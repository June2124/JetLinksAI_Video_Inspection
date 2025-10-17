# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
from typing import List, Dict, Any, Optional, Iterator, Tuple
from queue import Queue, Empty
import threading
from collections import deque

from src.utils import logger_utils

logger = logger_utils.get_logger(__name__)

# ---- DashScope 原生 SDK ----
try:
    import dashscope
    from dashscope import MultiModalConversation
    _HAS_MMC = True
except Exception as _e:
    logger.error("[B] 未检测到 DashScope SDK 或导入失败，请先：pip install dashscope ；error=%s", _e)
    _HAS_MMC = False


# ----------------- 路径与消息辅助 -----------------
def _is_http_url(p: str) -> bool:
    return isinstance(p, str) and p.lower().startswith(("http://", "https://"))


def _to_file_uri(path_or_uri: str) -> Optional[str]:
    """
    将本地路径统一为 file:// 绝对 URI；http(s)/已是 file:// 原样返回。
    Windows: file://D:/abs/path/file.mp4
    Linux/Mac: file:///abs/path/file.mp4
    """
    if not path_or_uri:
        return None
    if _is_http_url(path_or_uri) or path_or_uri.lower().startswith("file://"):
        return path_or_uri
    abs_path = os.path.abspath(path_or_uri).replace("\\", "/")
    return f"file://{abs_path}"


def _exists_local_from_uri(file_uri_or_path: str) -> bool:
    if not file_uri_or_path:
        return False
    if _is_http_url(file_uri_or_path):
        return True
    if file_uri_or_path.lower().startswith("file://"):
        local = file_uri_or_path[len("file://") :]
        return os.path.exists(local)
    return os.path.exists(file_uri_or_path)


def _build_msgs_for_video(video_uri: str, user_prompt: Optional[str]) -> List[Dict[str, Any]]:
    prompt = user_prompt or "请基于这段视频给出简洁的中文要点描述（事件、场景、角色、物体）。"
    return [{"role": "user", "content": [{"video": video_uri}, {"text": prompt}]}]


def _build_msgs_for_images(img_uris: List[str], user_prompt: Optional[str]) -> List[Dict[str, Any]]:
    prompt = user_prompt or "请根据这些关键帧总结该片段的主要内容（中文要点即可）。"
    content: List[Dict[str, Any]] = [{"text": prompt}]
    for u in img_uris:
        content.insert(0, {"image": u})
    return [{"role": "user", "content": content}]


def _q_put_with_retry(q: Queue, obj: Any, *, tries: int = 3, timeout: float = 0.5) -> bool:
    for i in range(1, tries + 1):
        try:
            q.put(obj, timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"[B] q_vlm.put 超时（第{i}/{tries}次）：{e}")
    return False


# ----------------- 模式提示词 -----------------
_DEFAULT_PROMPTS = {
    "online":  (
        os.getenv("VLM_USER_PROMPT_ONLINE")
        or "这是一段实时会议/直播的内容，请用中文要点总结：谁在说、讨论主题、任务分配、时间要点。若有屏幕分享或图表，请归纳关键信息。"
    ),
    "offline": (
        os.getenv("VLM_USER_PROMPT_OFFLINE")
        or "这是一段离线视频片段，请按时间顺序用中文要点总结主要事件、场景与物体；若出现显著画面变化，请突出说明。"
    ),
    "security": (
        os.getenv("VLM_USER_PROMPT_SECURITY")
        or "这是安防监控片段，请用中文要点报告：是否有人/车辆、入侵/徘徊/奔跑/打斗等异常、烟火/跌倒等风险；保守判定、避免过度臆测。"
    ),
}

def _resolve_prompt(global_prompt: Optional[str], mode_str: Optional[str]) -> Optional[str]:
    if global_prompt:
        return global_prompt
    key = (mode_str or "").strip().lower()
    return _DEFAULT_PROMPTS.get(key)


# ----------------- 重试 & 事件解析 -----------------
def _retry_stream(call_fn, *, max_tries: int, base: float):
    last_err = None
    for i in range(1, max_tries + 1):
        try:
            return call_fn()
        except Exception as e:
            last_err = e
            if i < max_tries:
                sleep_s = (base ** i) + min(0.6, 0.2 * i)
                logger.warning(f"[B] 流式创建失败({i}/{max_tries})：{e}；{sleep_s:.1f}s 后重试…")
                time.sleep(sleep_s)
    raise last_err


def _iter_mmc_events(stream_obj) -> Iterator[Tuple[Optional[str], Optional[dict]]]:
    """
    迭代 DashScope MMC 的流式响应，抽取：
    - delta_text: 当前增量文本
    - usage:      最近一次 usage
    """
    usage_cache: Optional[dict] = None
    for rsp in stream_obj:
        delta_text: Optional[str] = None
        try:
            out = rsp.get("output") or {}
            choices = out.get("choices") or []
            if choices:
                msg = (choices[0] or {}).get("message") or {}
                content = msg.get("content") or []
                parts: List[str] = []
                for it in content:
                    if isinstance(it, dict):
                        t = it.get("text")
                        if t:
                            parts.append(t)
                if parts:
                    delta_text = "".join(parts)
            u = rsp.get("usage") or {}
            if u:
                usage_cache = {
                    "prompt_tokens": u.get("input_tokens") or u.get("prompt_tokens"),
                    "completion_tokens": u.get("output_tokens") or u.get("completion_tokens"),
                    "total_tokens": u.get("total_tokens"),
                }
        except Exception:
            pass
        yield delta_text, usage_cache


# ----------------- 文本归一化与去重 -----------------
def _normalize_lines(s: str) -> List[str]:
    """
    把模型输出分解成“要点行”，用于和历史小结比对，尽量温和不过度修改原文。
    """
    if not s:
        return []
    raw_lines = [ln.strip() for ln in s.replace("\r", "").split("\n")]
    out: List[str] = []
    for ln in raw_lines:
        if not ln:
            continue
        # 去前缀符号
        for p in ("- ", "• ", "* ", "· "):
            if ln.startswith(p):
                ln = ln[len(p):].strip()
                break
        # 合并句末标点空格
        ln = ln.strip(" \u3000")
        if ln:
            out.append(ln)
    return out


def _join_as_bullets(lines: List[str]) -> str:
    return "\n".join("- " + ln for ln in lines)


# ----------------- 历史上下文拼装 -----------------
def _build_history_context_text(history: List[str], max_chars: int) -> str:
    """
    将最近若干段的小结拼成一段“历史小结”文本，用于放进 prompt。
    控制总体字符数上限（避免上下文过长）。
    """
    if not history:
        return ""
    # 近→远拼接，超过上限就截断
    buf = []
    remain = max_chars
    for one in reversed(history):
        if not one:
            continue
        t = one.strip()
        if not t:
            continue
        if len(t) + 1 > remain:
            break
        buf.append(t)
        remain -= (len(t) + 1)
    if not buf:
        return ""
    # 用分隔线清晰标记
    return "以下为“历史小结”（近到远，最多 N 段）：\n" + ("\n---\n".join(buf))


# ----------------- B 线程主体（带“只输出新增”能力） -----------------
def worker_b_vlm(q_video: Queue, q_vlm: Queue, q_ctrl: Queue, stop: object):
    running = False
    paused = False
    max_frames_cap: Optional[int] = None

    # 模型与提示词
    model_name = os.getenv("VLM_MODEL_NAME", "qwen3-vl-plus")
    global_user_prompt = os.getenv("VLM_USER_PROMPT", "").strip() or None
    current_mode: Optional[str] = None
    user_prompt: Optional[str] = _resolve_prompt(global_user_prompt, current_mode)

    # 新增：历史记忆&去重配置
    hist_max_rounds = int(os.getenv("VLM_CONTEXT_MEMORY_N", "30"))  # 记忆最多多少轮
    hist_max_chars  = int(os.getenv("VLM_CONTEXT_MAX_CHARS", "4000"))  # 放进 prompt 的历史字符上限
    history_deque: deque[str] = deque(maxlen=max(1, hist_max_rounds))  # 保存“已对外发布”的小结
    dedup_enabled = os.getenv("VLM_DEDUP_DONE", "1") == "1"  # done 收尾做去重（模型侧已约束，这里再保险）

    # 流式重试 & 心跳
    retry_times = int(os.getenv("VLM_STREAM_RETRY_TIMES", "2"))
    retry_base = float(os.getenv("VLM_STREAM_RETRY_BASE", "1.6"))
    hb_every = max(10, int(os.getenv("VLM_STOP_HEARTBEAT_N", "20")))

    # 可选：指定地域（例如新加坡）
    base_url = os.getenv("DASHSCOPE_BASE_HTTP_API_URL", "").strip()
    if base_url:
        try:
            dashscope.base_http_api_url = base_url
        except Exception:
            logger.warning("[B] 设置 DashScope base_http_api_url 失败，继续使用默认。")

    if not _HAS_MMC:
        logger.error("[B] DashScope MMC 不可用，将对每个片段直接输出错误收尾。")

    def _drain_all_ctrl_nonstop() -> bool:
        stop_flag = False
        while True:
            try:
                msg = q_ctrl.get_nowait()
            except Empty:
                break
            if msg is stop:
                stop_flag = True
                break
            if isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN"):
                stop_flag = True
                break
        return stop_flag

    def _poll_ctrl_heartbeat() -> bool:
        try:
            msg = q_ctrl.get_nowait()
        except Empty:
            return False
        if msg is stop:
            return True
        if isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN"):
            return True
        try:
            q_ctrl.put_nowait(msg)
        except Exception:
            pass
        return False

    # —— 只输出“新增”的系统/用户指令模板（可环境变量覆盖） —
    system_guardrail = os.getenv(
        "VLM_CONTEXT_SYSTEM",
        """仅当新片段出现了“可观察到的新增正向事实（新物体/动作/场景变化/文字/数字/图表/时间地点人物变化等）”时才输出；
        不要仅以“未发生/没有/未看到”作为新增要点。"""
    ).strip()

    user_guardrail_prefix = os.getenv(
        "VLM_CONTEXT_USER_PREFIX",
        "请在**参考历史小结**的前提下，只输出“历史中未出现的新信息/新细节/新动作/明显变化”。"
        "如果没有新增，请输出空字符串。输出尽量使用要点列表。"
        "你的回答格式必须放在一个段落内, 不要出现不同分段或换段落列要点等格式的输出。"
    ).strip()

    try:
        while True:
            # -------- 控制阶段 --------
            if _drain_all_ctrl_nonstop():
                logger.info("[B] 收到控制队列发送的STOP，退出")
                return

            if not running or paused:
                try:
                    msg = q_ctrl.get(timeout=0.2)
                except Empty:
                    continue
                if msg is stop:
                    logger.info("[B] 收到 控制队列发送的STOP，退出")
                    return
                if isinstance(msg, dict):
                    typ = msg.get("type")
                    if typ == "START":
                        running = True
                        paused = False
                        logger.info("[B] 收到 START，开始视觉解析")
                    elif typ == "RESUME":
                        running = True
                        paused = False
                        logger.info("[B] 收到 RESUME，继续视觉解析")
                    elif typ == "PAUSE":
                        paused = True
                        logger.info("[B] 收到 PAUSE，暂停视觉解析")
                    elif typ == "UPDATE_VLM_MAXFRAMES":
                        try:
                            max_frames_cap = int(msg.get("value"))
                            logger.info(f"[B] 收到UPDATE_VLM_MAXFRAMES，调整关键帧提取数量：{max_frames_cap}")
                        except Exception:
                            logger.warning("[B] UPDATE_VLM_MAXFRAMES 的 value 非法，忽略。")
                    elif typ == "MODE_CHANGE":
                        val = msg.get("value")
                        if isinstance(val, str) and val:
                            current_mode = val.strip().lower()
                            user_prompt = _resolve_prompt(global_user_prompt, current_mode)
                            logger.info(f"[B] 已切换模式为：{current_mode}；提示词来源={'GLOBAL' if global_user_prompt else 'BY_MODE'}")
                    elif typ in ("STOP", "SHUTDOWN"):
                        logger.info("[B] 收到控制队列发送的STOP，退出。")
                        return
                continue

            # -------- 取片段 --------
            try:
                item = q_video.get(timeout=0.1)
            except Empty:
                continue

            try:
                if item is stop:
                    logger.info("[B]收到数据队列的STOP，退出")
                    return
                if not isinstance(item, dict):
                    logger.warning("[B] 非法的 q_video 数据，忽略。")
                    continue

                seg_idx = int(item.get("segment_index", -1))
                small_video = item.get("small_video")
                keyframes: List[str] = item.get("keyframes") or []
                policy_used = (item.get("policy") or {}).get("policy_used")
                t0 = item.get("t0")
                t1 = item.get("t1")

                # ---- 构造消息（优先视频）----
                media_kind = None
                media_used: List[str] = []
                messages: List[Dict[str, Any]] = []

                if small_video:
                    vuri = _to_file_uri(small_video)
                    if vuri and _exists_local_from_uri(vuri):
                        media_kind = "video"
                        media_used = [vuri]
                        messages = _build_msgs_for_video(vuri, user_prompt)
                    else:
                        logger.warning(f"[B] seg#{seg_idx} small_video 文件不可用，退化到关键帧。")

                if not messages:
                    cap = max_frames_cap or 12
                    img_uris: List[str] = []
                    for p in keyframes[:cap]:
                        u = _to_file_uri(p)
                        if u and _exists_local_from_uri(u):
                            img_uris.append(u)
                    if not img_uris:
                        logger.warning(f"[B] seg#{seg_idx} 无可用的视频/图片 URI，跳过该段。")
                        continue
                    media_kind = "images"
                    media_used = img_uris
                    messages = _build_msgs_for_images(img_uris, user_prompt)

                # ===== 新增：附带历史小结到 prompt（从第2段起） =====
                history_text = _build_history_context_text(list(history_deque), hist_max_chars)
                if history_text:
                    # 用 system 提醒“只输出新增”
                    messages.insert(0, {"role": "system", "content": [{"text": system_guardrail}]})
                    # 在用户内容末尾再加一段“历史小结+只输出新增”的要求
                    messages.append({
                        "role": "user",
                        "content": [{"text": f"{user_guardrail_prefix}\n\n{history_text}"}]
                    })

                if not _HAS_MMC:
                    _q_put_with_retry(q_vlm, {
                        "type": "vlm_stream_done",
                        "segment_index": seg_idx,
                        "full_text": "[VLM_BACKEND_ERROR] DashScope SDK 不可用，请安装/升级 dashscope。",
                        "usage": None,
                        "model": model_name,
                        "latency_ms": 0,
                        "media_kind": "unknown",
                        "media_used": [],
                        "origin_policy": policy_used,
                        "t0": t0, "t1": t1,
                        # 兼容字段
                        "clip_t0": t0, "clip_t1": t1,
                        "frame_pts": [], "frame_indices": [],
                    })
                    continue

                # ---- 打开流并消费增量 ----
                t_start = time.time()
                full_content = ""
                usage = None
                seq = 1
                interrupted = False

                def _open_stream():
                    return MultiModalConversation.call(
                        model=model_name,
                        messages=messages,
                        stream=True,
                        incremental_output=True
                    )

                try:
                    stream_obj = _retry_stream(_open_stream, max_tries=retry_times, base=retry_base)
                    beat = 0

                    for delta, usage_part in _iter_mmc_events(stream_obj):
                        beat += 1

                        if delta:
                            full_content += delta
                            _q_put_with_retry(q_vlm, {
                                "type": "vlm_stream_delta",
                                "segment_index": seg_idx,
                                "delta": delta,
                                "seq": seq,
                                "model": model_name,
                                "media_kind": media_kind or "unknown",
                                # —— 时间/片段辅助信息（与 A 对齐）——
                                "media_ts": float(t0) + max(0.0, (float(t1) - float(t0)) / 2.0) if t0 is not None and t1 is not None else None,
                                "media_ts_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(
                                    (float(t0) + (float(t1)-float(t0))/2.0) if (t0 is not None and t1 is not None) else time.time()
                                )),
                                "clip_t0": t0, "clip_t1": t1,
                                "frame_pts": item.get("frame_pts") or [],
                                "frame_indices": item.get("frame_indices") or [],
                                "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
                                "origin_policy": policy_used,
                                "produce_ts": time.time(),
                                "produce_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                            })
                            seq += 1

                        if usage_part:
                            usage = usage_part

                        if (beat % hb_every) == 0 and _poll_ctrl_heartbeat():
                            interrupted = True
                            break

                except Exception as e:
                    logger.error(f"[B] VLM 流式调用异常（seg#{seg_idx}）：{e}")
                    _q_put_with_retry(q_vlm, {
                        "type": "vlm_stream_done",
                        "segment_index": seg_idx,
                        "full_text": f"[VLM_ERROR] {e}",
                        "usage": usage,
                        "model": model_name,
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "media_kind": media_kind or "unknown",
                        "media_used": media_used,
                        "origin_policy": policy_used,
                        "t0": t0, "t1": t1,
                        "clip_t0": t0, "clip_t1": t1,
                        "frame_pts": item.get("frame_pts") or [],
                        "frame_indices": item.get("frame_indices") or [],
                        "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
                        "produce_ts": time.time(),
                        "produce_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    })
                    continue

                # ---- 收尾（去重过滤，仅保留新增）----
                final_text = full_content
                suppressed = False
                if dedup_enabled and history_deque:
                    cur_lines = _normalize_lines(full_content)
                    hist_lines: List[str] = []
                    for h in history_deque:
                        hist_lines.extend(_normalize_lines(h))
                    hist_set = set(hist_lines)
                    new_lines = [ln for ln in cur_lines if ln not in hist_set]
                    final_text = _join_as_bullets(new_lines)
                    if not final_text.strip():
                        suppressed = True  # 没有新增

                # 记录用于后续轮的“历史小结”：只纳入对外最终发布的文本
                if final_text.strip():
                    history_deque.append(final_text)

                payload_done = {
                    "type": "vlm_stream_done",
                    "segment_index": seg_idx,
                    "full_text": final_text,
                    "usage": usage,
                    "model": model_name,
                    "latency_ms": int((time.time() - t_start) * 1000),
                    "media_kind": media_kind or "unknown",
                    "media_used": media_used,
                    "origin_policy": policy_used,
                    "t0": t0, "t1": t1,
                    # 兼容字段与时间辅助
                    "clip_t0": t0, "clip_t1": t1,
                    "frame_pts": item.get("frame_pts") or [],
                    "frame_indices": item.get("frame_indices") or [],
                    "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
                    "produce_ts": time.time(),
                    "produce_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime()),
                    # 新增：上下文信息（可用于调试）
                    "ctx_rounds": len(history_deque),
                    "suppressed_dup": bool(suppressed),
                }

                _q_put_with_retry(q_vlm, payload_done)

                if interrupted:
                    logger.info("[B] 已按 STOP 中断当前段落并退出。")
                    return

            finally:
                try:
                    q_video.task_done()
                except Exception:
                    pass

    finally:
        logger.info("[B] 线程退出清理完成。")
