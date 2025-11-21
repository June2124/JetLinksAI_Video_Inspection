# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import json, re, ast
from typing import List, Dict, Any, Optional, Tuple
from queue import Queue, Empty
from collections import deque

from src.utils import logger_utils
from src.utils.vlm_client.cloud_vlm_client import CloudVLMClient
from src.utils.vlm_client.local_sophon_vlm_client import LocalVLMClient
from src.configs.rtsp_batch_config import RTSPBatchConfig
from src.configs.vlm_config import VlmConfig
from src.all_enum import (
    MODEL,
    CLOUD_VLM_MODEL_NAME,
    LOCAL_VLM_MODEL_NAME,
    VLM_SYSTEM_PROMPT_PRESET,
    JSON_CONSTRAINT,
    VLM_DETECT_EVENT_LEVEL,
)

logger = logger_utils.get_logger(__name__)

# ----------------- 任务型严格 JSON Schema -----------------
EVENT_ARRAY_SCHEMA = {
    "type": "json_schema",
    "json_schema": {
        "name": "EventArray",
        "schema": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "describe": {"type": "string", "maxLength": 15},
                    "level": {"type": "string", "enum": ["NO_RISK", "LOW", "MEDIUM", "HIGH", "CRITICAL"]},
                    "suggestion": {"type": "string", "maxLength": 15},
                    "confidence": {"type": "number", "minimum": 0, "maximum": 1}
                },
                "required": ["type", "describe", "level", "suggestion", "confidence"],
                "additionalProperties": False
            }
        }
    }
}

_LEVEL_ORDER = {"NO_RISK":0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4 }
JSON_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

# ----------------- 文件与 URI 辅助 -----------------
def _is_http_url(p: str) -> bool:
    return isinstance(p, str) and p.lower().startswith(("http://", "https://"))

def _to_file_uri(path_or_uri: str) -> Optional[str]:
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
        local = file_uri_or_path[len("file://"):]
        return os.path.exists(local)
    return os.path.exists(file_uri_or_path)

def _export_evidence_list_to_static(
    evidence_images: List[str],
    seg_idx: int,
    vlm_config: Optional[VlmConfig] = None,
) -> List[str]:
    """
    将 evidence_images 列表中的本地文件导出到可由前端访问的静态目录，并返回对应 URL 列表。

    约定：
    - HTTP(S) URL：原样透传；
    - file:// 或 本地路径：复制到
        vlm_config.vlm_static_evidence_images_dir
      下，并按
        vlm_config.vlm_static_evidence_images_url_prefix
      生成 URL；
    - 若未提供 vlm_config，则回退到 ./static/evidence_images + /static/evidence_images。
    """
    out_urls: List[str] = []
    if not evidence_images:
        return out_urls

    # ---- 解析目录 & URL 前缀 ----
    if vlm_config is not None and getattr(vlm_config, "vlm_static_evidence_images_dir", None):
        static_dir = vlm_config.vlm_static_evidence_images_dir
    else:
        static_dir = os.path.join(os.getcwd(), "static", "evidence_images")

    if vlm_config is not None and getattr(vlm_config, "vlm_static_evidence_images_url_prefix", None):
        url_prefix = vlm_config.vlm_static_evidence_images_url_prefix
    else:
        url_prefix = "/static/evidence_images"

    url_prefix = url_prefix.rstrip("/")  # 防止双 //

    os.makedirs(static_dir, exist_ok=True)

    for i, uri in enumerate(evidence_images):
        # 1) 已经是 http(s) 的，直接透传
        if _is_http_url(uri):
            out_urls.append(uri)
            continue

        # 2) file:// 或本地路径
        local_path = _uri_to_local_path(uri) or uri
        if not os.path.exists(local_path):
            continue

        # 保留原始扩展名（默认 .jpg）
        ext = os.path.splitext(local_path)[1] or ".jpg"
        fname = f"seg{seg_idx:04d}_evdimg{i:02d}{ext}"
        dst = os.path.join(static_dir, fname)

        try:
            if os.path.abspath(local_path) != os.path.abspath(dst):
                import shutil
                shutil.copy2(local_path, dst)

            url = f"{url_prefix}/{fname}"
            out_urls.append(url)
        except Exception as e:
            logger.warning(f"[B] 导出证据帧失败：{e}")

    return out_urls



def _uri_to_local_path(file_uri: str) -> Optional[str]:
    if not file_uri:
        return None
    if file_uri.lower().startswith("file://"):
        return file_uri[len("file://"):]
    return None

# ----------------- Prompt 拼装 -----------------
def build_system_prompt(
    use_backend: str,
    vlm_system_prompt: str | VLM_SYSTEM_PROMPT_PRESET | None
) -> str:
    """
    - 若传入字符串：视为描述型，原样返回；
    - 若为预设枚举：
        * cloud 后端：原样返回预设；
        * local 后端：预设 + JSON 约束；
    - 若为空：返回空字符串（由上游自行控制）。
    """
    if not vlm_system_prompt:
        return ""

    if isinstance(vlm_system_prompt, str):
        return vlm_system_prompt.strip()

    if isinstance(vlm_system_prompt, VLM_SYSTEM_PROMPT_PRESET):
        base = vlm_system_prompt.value.strip()
        if use_backend == "local":
            json_constraint_text = JSON_CONSTRAINT.JSON_CONSTRAINT.value.strip()
            return f"{base}\n\n{json_constraint_text}"
        else:
            return base

    return ""

def _build_msgs_for_video(video_uri: str, user_prompt: Optional[str]) -> List[Dict[str, Any]]:
    prompt = user_prompt or "请根据视频判断是否出现预定义事件，并输出 JSON。"
    return [{"role": "user", "content": [{"video": video_uri}, {"text": prompt}]}]

def _pair_keyframes_with_pts(
    keyframes: List[str],
    frame_pts: List[float],
    *,
    cap: Optional[int] = None,
) -> List[Tuple[str, float]]:
    n = min(len(keyframes), len(frame_pts))
    if len(keyframes) != len(frame_pts):
        logger.warning(
            "[B] keyframes (%d) 与 frame_pts (%d) 长度不一致，按最短长度(%d)截断。",
            len(keyframes), len(frame_pts), n
        )
    pairs = [(keyframes[i], float(frame_pts[i])) for i in range(n)]
    if cap is not None:
        pairs = pairs[:cap]
    return pairs

def _build_msgs_images_only(
    pairs: List[Tuple[str, float]],
    *,
    is_task: bool,
) -> List[Dict[str, Any]]:
    content: List[Dict[str, Any]] = []
    for p, _t in pairs:
        u = _to_file_uri(p)
        if u and _exists_local_from_uri(u):
            content.append({"image": u})
    if not content:
        return []
    return [{"role": "user", "content": content}]

# ----------------- STOP 感知 & 安全投递 -----------------
def _ctrl_stop_requested(q_ctrl: Queue | None, stop: object | None) -> bool:
    if q_ctrl is None or stop is None:
        return False
    try:
        while True:
            msg = q_ctrl.get_nowait()
            if (msg is stop) or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                logger.warning("[B] 检测到控制队列 STOP 哨兵, 停止本次投递")
                return True
            try:
                q_ctrl.put_nowait(msg)
            except Exception:
                pass
            return False
    except Empty:
        return False

def _q_put_with_retry(
    q: Queue,
    obj: Any,
    *,
    tries: int = 3,
    timeout: float = 0.5,
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
    drop_on_stop: bool = True,
    drop_on_timeout: bool = True,
) -> bool:
    if _ctrl_stop_requested(q_ctrl, stop):
        return not drop_on_stop
    for i in range(1, tries + 1):
        try:
            q.put(obj, timeout=timeout)
            return True
        except Exception as e:
            logger.warning(f"[B] q_vlm.put 超时（第{i}/{tries}次）：{e}")
            if _ctrl_stop_requested(q_ctrl, stop):
                return not drop_on_stop
    if drop_on_timeout:
        return False
    try:
        q.put(obj, timeout=0.2)
        return True
    except Exception:
        return False

# ----------------- 文本去重/归一化（仅描述型使用） -----------------
def _normalize_lines(s: str) -> List[str]:
    if not s:
        return []
    raw_lines = [ln.strip() for ln in s.replace("\r", "").split("\n")]
    out = []
    for ln in raw_lines:
        if not ln:
            continue
        for p in ("- ", "• ", "* ", "· "):
            if ln.startswith(p):
                ln = ln[len(p):].strip()
                break
        ln = ln.strip(" \u3000")
        if ln:
            out.append(ln)
    return out

def _join_as_bullets(lines: List[str]) -> str:
    return "\n".join("- " + ln for ln in lines)

def _build_history_context_text(history: List[str], max_chars: int) -> str:
    if not history:
        return ""
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
    return "以下为“历史小结”（近到远，最多 N 段）：\n" + ("\n---\n".join(buf))

# ----------------- 事件发包 -----------------
def _emit_delta(
    q_vlm: Queue,
    seg_idx: int,
    delta: str,
    seq: int,
    model: str,
    item: dict,
    *,
    streaming: bool,
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
) -> None:
    _q_put_with_retry(
        q_vlm,
        {
            "type": "vlm_stream_delta",
            "segment_index": seg_idx,
            "delta": delta,
            "seq": seq,
            "model": model,
            "streaming": bool(streaming),
            "produce_ts": time.time(),
            # 时间轴/帧
            "clip_t0": item.get("t0"),
            "clip_t1": item.get("t1"),
            "frame_pts": item.get("frame_pts") or [],
            "frame_indices": item.get("frame_indices") or [],
            "t0_iso": item.get("t0_iso"),
            "t1_iso": item.get("t1_iso"),
            "t0_epoch": item.get("t0_epoch"),
            "t1_epoch": item.get("t1_epoch"),
            "frame_epoch": item.get("frame_epoch") or [],
            "frame_iso": item.get("frame_iso") or [],
            # 编码策略
            "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
            "origin_policy": (item.get("policy") or {}).get("policy_used"),
            # 透传流元信息
            "stream_url": item.get("stream_url"),
            "stream_index": item.get("stream_index"),
            "stream_segment_index": item.get("stream_segment_index"),
            "window_index_in_stream": item.get("window_index_in_stream"),
            "polling_round_index": item.get("polling_round_index"),
        },
        q_ctrl=q_ctrl,
        stop=stop,
        drop_on_stop=True,
        drop_on_timeout=True,
    )

def _emit_done(
    q_vlm: Queue,
    seg_idx: int,
    full_text: str,
    model: str,
    item: dict,
    *,
    usage: dict | None,
    latency_ms: int,
    streaming: bool,
    suppressed_dup: bool | None = None,
    ctx_rounds: int | None = None,
    evidence_images: Optional[List[str]] = None,
    evidence_image_urls: Optional[List[str]] = None,
    q_ctrl: Optional[Queue] = None,
    stop: object = None,
) -> None:
    payload = {
        "type": "vlm_stream_done",
        "segment_index": seg_idx,
        "full_text": full_text or "",
        "usage": usage,
        "model": model,
        "streaming": bool(streaming),
        "latency_ms": latency_ms,
        "produce_ts": time.time(),
        # 时间轴/帧
        "clip_t0": item.get("t0"),
        "clip_t1": item.get("t1"),
        "frame_pts": item.get("frame_pts") or [],
        "frame_indices": item.get("frame_indices") or [],
        "t0_iso": item.get("t0_iso"),
        "t1_iso": item.get("t1_iso"),
        "t0_epoch": item.get("t0_epoch"),
        "t1_epoch": item.get("t1_epoch"),
        "frame_epoch": item.get("frame_epoch") or [],
        "frame_iso": item.get("frame_iso") or [],
        # 透传去重/上下文/证据
        "suppressed_dup": bool(suppressed_dup) if suppressed_dup is not None else None,
        "ctx_rounds": ctx_rounds,
        "evidence_images": evidence_images or [],
        "evidence_image_urls": evidence_image_urls or [],
        # 编码策略
        "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
        "origin_policy": (item.get("policy") or {}).get("policy_used"),
        # 透传流元信息
        "stream_url": item.get("stream_url"),            # 当前片段所属的 RTSP 地址（唯一标识一条流）
        "stream_index": item.get("stream_index"),        # 当前流在轮询列表中的索引位置（0-based）
        "stream_segment_index": item.get("stream_segment_index"),  # 该流自任务启动以来的连续窗口序号（全局计数）
        "window_index_in_stream": item.get("window_index_in_stream"), # 该流在“当前轮”中是第几个窗口（局部计数）
        "polling_round_index": item.get("polling_round_index"),    # 当前是第几轮轮询（SECURITY_POLLING 模式下从 0 开始计数）
    }
    _q_put_with_retry(
        q_vlm,
        payload,
        q_ctrl=q_ctrl,
        stop=stop,
        drop_on_stop=True,
        drop_on_timeout=True,
    )

# ----------------- JSON 提取/解析（任务型） -----------------
def _extract_json_str(s: str) -> str:
    if not s:
        return ""
    s = s.replace("\ufeff", "").strip()

    m = JSON_FENCE_RE.search(s)
    if m:
        return m.group(1).strip()

    start = None
    for i, ch in enumerate(s):
        if ch in "[{]":
            start = i
            break
    if start is None:
        return ""

    stack = []
    end = None
    for i in range(start, len(s)):
        ch = s[i]
        if ch in "[{]":
            stack.append(ch)
        elif ch in "]}":
            if not stack:
                break
            left = stack.pop()
            if (left, ch) not in (('[', ']'), ('{', '}')):
                return ""
            if not stack:
                end = i + 1
                break
    if not end:
        return ""

    candidate = s[start:end].strip()
    candidate = re.sub(r",(\s*[\]}])", r"\1", candidate)
    return candidate

def _loads_relaxed_json(s: str) -> Any:
    if not s:
        raise ValueError("empty json string")
    try:
        return json.loads(s)
    except Exception:
        pass

    s2 = s
    s2 = re.sub(r"(?<!\\)'", '"', s2)
    s2 = s2.replace(" True", " true").replace(" False", " false").replace(" None", " null")
    try:
        return json.loads(s2)
    except Exception:
        pass

    try:
        return ast.literal_eval(s)
    except Exception:
        raise ValueError("cannot parse json")

def parse_vlm_events_strict(text: str) -> list[dict]:
    raw = _extract_json_str(text)
    if not raw:
        raise ValueError("no json block found")
    obj = _loads_relaxed_json(raw)
    if isinstance(obj, dict):
        obj = [obj]
    if not isinstance(obj, list):
        raise ValueError("json is not list/dict")

    out: list[dict] = []
    for it in obj:
        if isinstance(it, dict):
            out.append(it)
        else:
            out.append({"type": "UNKNOWN", "describe": str(it), "level": "LOW", "suggestion": "", "confidence": 0.0})
    return out

def filter_events_by_level_list(events: list[dict], min_level: Optional[str]) -> tuple[list[dict], Optional[str]]:
    if not min_level or min_level not in _LEVEL_ORDER:
        return events, None
    keep: list[dict] = []
    max_lv = None
    max_val = -1
    th = _LEVEL_ORDER[min_level]
    for it in events:
        lv = str(it.get("level", "")).upper()
        v = _LEVEL_ORDER.get(lv, -1)
        if v >= th:
            keep.append(it)
            if v > max_val:
                max_val, max_lv = v, lv
    return keep, max_lv

# ----------------- 入口（B 线程主体） -----------------
def worker_b_vlm(
    q_video: Queue,
    q_vlm: Queue,
    q_ctrl: Queue,
    stop: object,
    model: MODEL,
    rtsp_batch_config: Optional[RTSPBatchConfig] = None,
    vlm_config: Optional[VlmConfig] = None,
):
    """
    B 线程（VLM 视觉语义解析）
    - is_task：SECURITY_SINGLE / SECURITY_POLLING → True；OFFLINE → False
    - OFFLINE：提示词来自 VlmConfig.offline_system_prompt
    - SECURITY_*：提示词来自 A 侧 payload.rtsp_system_prompt（缺失再按 stream_url 兜底到 rtsp_batch_config）
    - 任务型：非流式 + json_schema；完成后做 level 过滤
    - 非任务型：允许流式；做简单去重与历史拼接
    """
    running = False
    paused = False

    vlm_config = vlm_config or VlmConfig()
    use_backend = vlm_config.vlm_backend
    if use_backend == "cloud" and not isinstance(vlm_config.vlm_cloud_model_name, CLOUD_VLM_MODEL_NAME):
        raise TypeError("vlm_cloud_model_name 必须是 CLOUD_VLM_MODEL_NAME 枚举实例")
    if use_backend == "local" and not isinstance(vlm_config.vlm_local_model_name, LOCAL_VLM_MODEL_NAME):
        raise TypeError("vlm_local_model_name 必须是 LOCAL_VLM_MODEL_NAME 枚举实例")
    model_name = (vlm_config.vlm_local_model_name.value
                  if use_backend == "local"
                  else vlm_config.vlm_cloud_model_name.value)

    # 按模式判断任务型
    is_task = model in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING)

    # 任务型是否拼历史（默认不拼，除非开启）
    task_history_enabled = bool(getattr(vlm_config, "vlm_task_history_enabled", False))

    # 最小等级筛选
    raw_min_level = getattr(vlm_config, "vlm_event_min_level", None)
    if isinstance(raw_min_level, VLM_DETECT_EVENT_LEVEL):
        event_min_level: Optional[str] = raw_min_level.name
    elif isinstance(raw_min_level, str):
        event_min_level = raw_min_level.strip().upper()
    else:
        event_min_level = None
    if event_min_level and event_min_level not in _LEVEL_ORDER:
        logger.warning("[B] 配置的 vlm_event_min_level=%r 无效，将忽略筛选。", raw_min_level)
        event_min_level = None

    # 最大关键帧数量（可选）
    if use_backend == "cloud":
        max_frames_cap: Optional[int] = getattr(vlm_config, "vlm_max_frames", 1)
    else:
        max_frames_cap: Optional[int] = 1  # 本地模型仅支持单图

    # 历史上下文存储 & 去重（仅描述型开启）
    if is_task:
        dedup_enabled = False
        if task_history_enabled:
            hist_max_rounds = 30
            hist_max_chars = 4000
            history_deque: deque[str] = deque(maxlen=hist_max_rounds)
        else:
            hist_max_rounds = 0
            hist_max_chars = 0
            history_deque = deque(maxlen=0)
    else:
        dedup_enabled = True
        hist_max_rounds = 30
        hist_max_chars = 4000
        history_deque: deque[str] = deque(maxlen=hist_max_rounds)

    # 主循环
    try:
        while True:
            if not running or paused:
                try:
                    msg = q_ctrl.get(timeout=0.2)
                except Empty:
                    continue
                if msg is stop:
                    logger.info("[B] 收到 STOP，退出")
                    return
                if isinstance(msg, dict):
                    typ = msg.get("type")
                    if typ in ("START", "RESUME"):
                        running, paused = True, False
                        logger.info("[B] 启动视觉解析")
                    elif typ == "PAUSE":
                        paused = True
                        logger.info("[B] 暂停视觉解析")
                    elif typ == "STOP":
                        logger.info("[B] 收到 STOP，退出。")
                        return
                continue

            # 取片段
            try:
                item = q_video.get(timeout=0.1)
            except Empty:
                continue

            try:
                if item is stop:
                    logger.info("[B] 收到数据队列 STOP，退出")
                    return

                seg_idx = int(item.get("segment_index", -1))
                small_video = item.get("small_video")
                keyframes: List[str] = item.get("keyframes") or []
                frame_pts: List[float] = item.get("frame_pts") or []

                # -------- 流元信息, 用于日志记录和透传 --------
                if model in(MODEL.SECURITY_POLLING,MODEL.SECURITY_SINGLE):
                    stream_url = item.get("stream_url")
                    stream_index = item.get("stream_index")
                    stream_segment_index = item.get("stream_segment_index")
                    polling_round_index = item.get("polling_round_index")

                    logger.info(
                        "[B] 正在处理片段：seg#%d | 流索引=%s | 流地址=%s | 当前流切片序号=%s | 轮询轮次=%s",
                        seg_idx,
                        stream_index if stream_index is not None else "-",
                        stream_url or "-",
                        stream_segment_index if stream_segment_index is not None else "-",
                        polling_round_index if polling_round_index is not None else "-"
                    )

                # -------- 计算 system prompt（按模式与每个片段独立决定） --------
                if model == MODEL.OFFLINE:
                    sys_text = build_system_prompt(use_backend, getattr(vlm_config, "offline_system_prompt", "") or "")
                    # OFFLINE 允许流式
                    want_streaming = bool(vlm_config.vlm_streaming)
                    response_fmt = None
                else:
                    # SECURITY_*：先用 A 侧 payload 的 rtsp_system_prompt；缺失才兜底到 batch_config 里按 stream_url 找
                    rtsp_sys_prompt = item.get("rtsp_system_prompt", None)
                    if rtsp_sys_prompt is None and rtsp_batch_config and getattr(item, "get", None):
                        try:
                            stream_url = item.get("stream_url")
                            if stream_url and rtsp_batch_config and rtsp_batch_config.polling_list:
                                for it in rtsp_batch_config.polling_list:
                                    if getattr(it, "rtsp_url", None) == stream_url:
                                        rtsp_sys_prompt = getattr(it, "rtsp_system_prompt", None) or getattr(it, "rtsp_prompt", None)
                                        break
                        except Exception:
                            pass
                    sys_text = build_system_prompt(use_backend, rtsp_sys_prompt)
                    # 任务型：强制非流式 + json_schema
                    want_streaming = False
                    response_fmt = EVENT_ARRAY_SCHEMA

                # -------- 构造 VLM 输入：优先关键帧；否则回退小视频 --------
                messages: List[Dict[str, Any]] = []
                evidence_images: List[str] = []
                evidence_image_urls: List[str] = []

                if keyframes and frame_pts:
                    pairs = _pair_keyframes_with_pts(keyframes, frame_pts, cap=max_frames_cap)
                    if not pairs:
                        logger.warning(f"[B] seg#{seg_idx} keyframes/frame_pts 为空或未配对，回退到 small_video。")
                    else:
                        messages = _build_msgs_images_only(pairs, is_task=is_task)
                        # 证据：全量 keyframes（转 file://） 当前默认所有keyfream都是evidence_image
                        evidence_images = [(_to_file_uri(p) or p) for p, _ in pairs if (_to_file_uri(p) or p)]
                        evidence_images = [u for u in evidence_images if _exists_local_from_uri(u) or _is_http_url(u)]
                        # 导出静态 URL
                        evidence_image_urls = _export_evidence_list_to_static(evidence_images, seg_idx, vlm_config)

                if not messages and small_video:
                    vuri = _to_file_uri(small_video)
                    if vuri and _exists_local_from_uri(vuri):
                        messages = _build_msgs_for_video(vuri, user_prompt=None)

                if not messages:
                    logger.warning(f"[B] seg#{seg_idx} 无可用的视频/图片 URI，跳过该段。")
                    continue

                # 在消息最前插入 system prompt（如为空则不插入）
                if sys_text.strip():
                    messages.insert(0, {"role": "system", "content": [{"text": sys_text}]})

                # user 级历史上下文（任务型默认不拼，除非显式开启）
                if (not is_task) or (is_task and task_history_enabled):
                    history_text = _build_history_context_text(list(history_deque), 4000)
                    if history_text:
                        messages.append({"role": "user", "content": [{"text": history_text}]})

                # ---- 调用后端 ----
                logger.info(f'[B] seg#{seg_idx}, 最终 system prompt 为：\n{sys_text}')
                logger.info(f'[B] seg#{seg_idx}, 完整 messages：\n{json.dumps(messages, ensure_ascii=False, indent=2)}')

                t_start = time.time()
                if use_backend == "cloud":
                    try:
                        import dashscope  # noqa: F401
                        from dashscope import MultiModalConversation  # noqa: F401
                        _HAS_MMC = True
                    except Exception as _e:
                        raise RuntimeError(f"[B] 未检测到 DashScope SDK 或导入失败，请先：pip install dashscope ；error={_e}")
                    cloud_vlm_client = CloudVLMClient(
                        model_name=model_name,
                        messages=messages,
                        want_streaming=want_streaming,
                        response_format=response_fmt,
                        HAS_MMC=_HAS_MMC,
                        q_ctrl=q_ctrl,
                        stop=stop,
                    )
                    try:
                        mode_used, iter_pair, nonstream_pair = cloud_vlm_client.infer()
                    except Exception as e:
                        logger.error(f"[B] VLM 调用失败（seg#{seg_idx}）：{e}")
                        _emit_done(
                            q_vlm, seg_idx,
                            full_text=f"[VLM_BACKEND_ERROR] {e}",
                            model=model_name, item=item,
                            usage=None,
                            latency_ms=int((time.time() - t_start) * 1000),
                            streaming=False,
                            suppressed_dup=None,
                            ctx_rounds=len(history_deque),
                            evidence_images=evidence_images,
                            evidence_image_urls=evidence_image_urls,
                            q_ctrl=q_ctrl, stop=stop
                        )
                        continue
                else:
                    local_vlm_client = LocalVLMClient(
                        model_name=model_name,
                        messages=messages,
                        q_ctrl=q_ctrl,
                        stop=stop,
                    )
                    try:
                        mode_used, iter_pair, nonstream_pair = local_vlm_client.infer()
                    except Exception as e:
                        logger.error(f"[B] 本地VLM 调用失败（seg#{seg_idx}）：{e}")
                        _emit_done(
                            q_vlm, seg_idx,
                            full_text=f"[LOCAL_VLM_ERROR] {e}",
                            model=model_name, item=item,
                            usage=None,
                            latency_ms=int((time.time() - t_start) * 1000),
                            streaming=False,
                            suppressed_dup=None,
                            ctx_rounds=len(history_deque),
                            evidence_images=evidence_images,
                            evidence_image_urls=evidence_image_urls,
                            q_ctrl=q_ctrl, stop=stop
                        )
                        continue

                # ---- 统一处理结果 ----
                usage = None
                final_text = ""
                suppressed = False
                if mode_used == "stream":
                    seq = 1
                    buf: List[str] = []
                    for delta, usage_part in iter_pair:  # type: ignore
                        if delta:
                            buf.append(delta)
                            _emit_delta(
                                q_vlm, seg_idx, delta, seq, model_name, item,
                                streaming=True, q_ctrl=q_ctrl, stop=stop
                            )
                            seq += 1
                        if usage_part:
                            usage = usage_part
                    final_text = "".join(buf)
                    streaming_flag = True
                else:
                    final_text, usage = nonstream_pair or ("", None)  # type: ignore
                    streaming_flag = False

                status = (usage or {}).get("status")
                out_text = final_text
                if is_task:
                    # --- 本地 VLM 返回非 OK 状态，直接认为该段没有有效事件 ---
                    if status and status != "ok":
                        logger.warning(
                            "[B] seg#%d 本地VLM返回非OK状态: %s, full_text 前 100 字: %s",
                            seg_idx, status, (final_text or "")[:100]
                        )
                        out_text = "[]"           # 给前端一个明确“无事件”的结果
                        suppressed = False
                    else:
                        try:
                            events_list = parse_vlm_events_strict(final_text)
                            if event_min_level:
                                events_list, _max_lv = filter_events_by_level_list(events_list, event_min_level)
                            out_text = json.dumps(events_list, ensure_ascii=False) if events_list else "[]"
                        except Exception as e:
                            logger.warning("[B] seg#%d 事件解析失败（保持原样透传）：%s", seg_idx, e)
                            try:
                                os.makedirs("out", exist_ok=True)
                                with open("out/last_vlm_raw.txt", "w", encoding="utf-8") as f:
                                    f.write(final_text or "")
                            except Exception:
                                pass
                        suppressed = False
                else:
                    if history_deque and dedup_enabled:
                        cur_lines = _normalize_lines(final_text)
                        hist_lines = [ln for h in history_deque for ln in _normalize_lines(h)]
                        new_lines = [ln for ln in cur_lines if ln not in set(hist_lines)]
                        out_text = _join_as_bullets(new_lines)
                        suppressed = (not out_text.strip())
                        if out_text.strip():
                            history_deque.append(out_text)

                _emit_done(
                    q_vlm, seg_idx,
                    full_text=out_text,
                    model=model_name, item=item,
                    usage=usage,
                    latency_ms=int((time.time() - t_start) * 1000),
                    streaming=streaming_flag,
                    suppressed_dup=suppressed,
                    ctx_rounds=len(history_deque),
                    evidence_images=evidence_images,
                    evidence_image_urls=evidence_image_urls,
                    q_ctrl=q_ctrl, stop=stop
                )

            finally:
                try:
                    q_video.task_done()
                except Exception:
                    pass
    finally:
        logger.info("[B] 线程退出清理完成。")
