# -*- coding: utf-8 -*-
from __future__ import annotations

from typing import Optional, Tuple, List, Dict, Any, Callable
from queue import Queue, Empty, Full
from time import sleep, time as now_ts
from datetime import datetime, timezone
import numpy as np
import cv2
import os

from src.all_enum import MODEL
from src.configs.rtsp_batch_config import RTSPBatchConfig, RTSP
from src.configs.cut_config import CutConfig
from src.utils.logger_utils import get_logger

from src.utils.ffmpeg.python_ffmpeg_utils import (
    probe_duration_seconds,
    cut_and_standardize_segment,
    grab_frame_by_index,
)

from src.utils.opencv.python_opencv_utils import (
    imwrite_jpg,
    get_video_meta,
    resize_keep_w,
    ssim_gray,
    bgr_ratio_score,
)

logger = get_logger(__name__)

RESIZE_W     = 320
EXPECT_CANDS = 16
OUT_DIR      = "static/out"


# ===================== 小工具 =====================
def _file_exists_nonzero(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


def _epoch_to_iso_utc(ts_epoch: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc) \
                       .isoformat(timespec="seconds") \
                       .replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _safe_put_with_ctrl(
    q: Optional[Queue],
    obj: dict,
    q_ctrl: Queue,
    stop: object,
    *,
    timeout: float = 0.2,
    max_tries: int = 50
) -> bool:
    """生产者安全投递，期间优先响应 STOP 控制。"""
    if q is None:
        return False
    tries = 0
    while True:
        # 优先处理 STOP
        try:
            msg = q_ctrl.get_nowait()
            if msg is stop or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                logger.info("[A] 检测到 STOP, 放弃投递。")
                return False
            try:
                q_ctrl.put_nowait(msg)
            except Full:
                pass
        except Empty:
            pass

        try:
            q.put(obj, timeout=timeout)
            return True
        except Full:
            tries += 1
            if tries >= max_tries:
                logger.warning("[A] 队列拥堵，放弃投递。")
                return False
        except Exception as e:
            logger.warning(f"[A] put 异常：{e}")
            return False


def _drain_queue_completely(q: Optional[Queue], max_batch: int = 200000) -> int:
    """尽最大努力丢弃队列中全部元素，返回丢弃个数。"""
    if q is None:
        return 0
    dropped = 0
    for _ in range(max_batch):
        try:
            q.get_nowait()
            dropped += 1
        except Empty:
            break
        except Exception:
            break
    if dropped:
        logger.info("[A] 已清空下游队列：丢弃 %d 条旧消息。", dropped)
    return dropped


# ===================== 关键帧选择 =====================
def pick_best_change_frame_with_pts(
    video_path: str,
    out_dir: str,
    tag: str,
    *,
    seg_t0: float,
    alpha_bgr: float = 0.5,
    topk_frames: int = 1
) -> Tuple[List[str], List[float], List[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("[A] 无法打开视频, 准备回退到固定抽1帧。")
        return [], [], []

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        stride = max(1, (total // EXPECT_CANDS) if total > 0 else int(round(fps / 4)))

        prev_small = None
        prev_gray = None
        candidates: List[Tuple[float, int, float, np.ndarray]] = []

        while True:
            for _ in range(stride - 1):
                if not cap.grab():
                    break
            ret, frame = cap.read()
            if not ret:
                break

            cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            small = resize_keep_w(frame, RESIZE_W)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

            if prev_small is not None and prev_gray is not None:
                br = bgr_ratio_score(prev_small, small)  # 0~1 越大越不同
                sv = ssim_gray(prev_gray, gray)          # 0~1 越大越相似
                score = alpha_bgr * br + (1.0 - alpha_bgr) * (1.0 - sv)
            else:
                score = -1.0

            pts = float(seg_t0) + (cur_idx / fps)
            candidates.append((score, cur_idx, pts, frame.copy()))
            prev_small, prev_gray = small, gray

        if not candidates:
            return [], [], []

        candidates.sort(key=lambda x: x[0], reverse=True)
        k = max(1, int(topk_frames) if topk_frames else 1)
        selected = candidates[:k]
        selected.sort(key=lambda x: x[1])

        os.makedirs(out_dir, exist_ok=True)
        paths: List[str] = []
        pts_list: List[float] = []
        idxs: List[int] = []

        for rank, (_score, idx, pts, frame_bgr) in enumerate(selected):
            out_path = os.path.join(out_dir, f"{tag}_kf_{rank:02d}_n{idx:06d}.jpg")
            imwrite_jpg(out_path, frame_bgr, quality=90)
            paths.append(out_path)
            pts_list.append(pts)
            idxs.append(idx)

        return paths, pts_list, idxs

    finally:
        cap.release()


def sample_one_frame_with_pts(
    video_path: str,
    out_dir: str,
    tag: str,
    *,
    seg_t0: float
) -> Tuple[List[str], List[float], List[int]]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        fps, total = get_video_meta(video_path)
        if total <= 0:
            return [], [], []
        outs = grab_frame_by_index(video_path, out_dir, f"{tag}_snap_0000", total // 2)
        if not outs:
            return [], [], []
        if fps <= 0:
            fps = 25.0
        return outs[:1], [float(seg_t0) + (total // 2) / fps], [total // 2]

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        if fps <= 0:
            fps = 25.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0:
            cap.release()
            return [], [], []

        target_idx = total // 2
        cur_idx = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            cur_idx += 1
            if cur_idx == target_idx:
                out_path = os.path.join(out_dir, f"{tag}_snap_0000.jpg")
                imwrite_jpg(out_path, frame, quality=90)
                return [out_path], [float(seg_t0) + (cur_idx / fps)], [cur_idx]
    finally:
        cap.release()

    outs = grab_frame_by_index(video_path, out_dir, f"{tag}_snap_0000", target_idx)
    if not outs:
        return [], [], []
    return outs[:1], [float(seg_t0) + (target_idx / fps)], [target_idx]


def get_now_wait_formatted_time(timestamp: float) -> Tuple[str, str]:
    # timestamp 为预计下一轮循环开始的时间戳
    import datetime
    now_date_obj = datetime.datetime.now()
    wait_date_obj = datetime.datetime.fromtimestamp(timestamp)
    now_date_formatted = now_date_obj.strftime("%Y-%m-%d %H:%M:%S")
    wait_date_formatted = wait_date_obj.strftime("%Y-%m-%d %H:%M:%S")
    return now_date_formatted, wait_date_formatted


# ===================== 主线程（支持 OFFLINE / 单流RTSP / 轮询RTSP + 动态增删改流 & 动态改间隔） =====================
def worker_a_cut(
    url: Optional[str],
    rtsp_batch_config: Optional[RTSPBatchConfig],
    have_audio_track: bool,
    mode: MODEL,
    q_audio: Optional[Queue],
    q_video: Optional[Queue],
    q_ctrl: Queue,
    stop: object,
    cut_config: CutConfig = None,
):
    running = False
    paused = False
    cur_mode = mode

    # —— 恢复后是否需要 RTSP 跳直播点 —— #
    resume_jump_to_live = False

    # ---- 解析配置 ----
    cut_config = cut_config or CutConfig()
    cut_window_sec = float(cut_config.cut_window_sec)
    alpha_bgr = float(cut_config.alpha_bgr)
    topk_frames = int(cut_config.topk_frames)
    logger.info(
        f"[A] 切片配置：窗口 {cut_window_sec}s，alpha_bgr={alpha_bgr}, "
        f"topK={topk_frames}, has_audio_track={have_audio_track}"
    )

    # 离线总时长（RTSP/直播 → None）
    def _probe_duration(u: Optional[str]) -> Optional[float]:
        if not u:
            return None
        try:
            return probe_duration_seconds(u)
        except Exception:
            return None

    # ---------- 控制消息处理 ----------
    def _handle_add_stream(msg: Dict[str, Any], cur_list: List[Any]) -> None:
        """动态增加一条流：兼容 {type:'ADD_STREAM', stream:{...}} / {type:'RTSP_ADD_STREAM', item:{...}}"""
        st = msg.get("stream")
        if st is None:
            st = msg.get("item")
        if st is None:
            logger.warning("[A] ADD_STREAM 缺少 'stream' 或 'item'")
            return
        try:
            if isinstance(st, RTSP):
                new_item = st
            elif isinstance(st, dict):
                new_item = RTSP(**st)
            else:
                logger.warning("[A] ADD_STREAM 非法类型：%r", type(st))
                return
        except Exception as e:
            logger.warning("[A] ADD_STREAM 构造 RTSP 失败：%s", e)
            return

        # 轮询模式下：最多 50 路，和主控一致
        if cur_mode == MODEL.SECURITY_POLLING and len(cur_list) >= 50:
            logger.warning(
                "[A] ADD_STREAM 被拒：SECURITY_POLLING 模式最多 50 路（当前=%d）",
                len(cur_list)
            )
            return

        # 去重（按 url）
        urls = [it.rtsp_url for it in cur_list]
        if new_item.rtsp_url in urls:
            logger.info("[A] ADD_STREAM 已存在：%s（忽略）", new_item.rtsp_url)
            return
        cur_list.append(new_item)
        logger.info("[A] ADD_STREAM 成功：%s（当前共 %d 路）", new_item.rtsp_url, len(cur_list))

    def _handle_remove_stream(msg: Dict[str, Any], cur_list: List[Any]) -> None:
        """动态删除一条流：兼容 {type:'REMOVE_STREAM'| 'RTSP_REMOVE_STREAM', rtsp_url:'...'}"""
        u = msg.get("rtsp_url")
        if not u:
            logger.warning("[A] REMOVE_STREAM 缺少 'rtsp_url'")
            return

        before = len(cur_list)
        # 轮询模式下：至少保留 2 路，与主控一致
        if cur_mode == MODEL.SECURITY_POLLING and before <= 2:
            logger.warning(
                "[A] REMOVE_STREAM 被拒：SECURITY_POLLING 模式下需要至少 2 路（当前=%d）",
                before
            )
            return

        cur_list[:] = [it for it in cur_list if it.rtsp_url != u]
        after = len(cur_list)
        if after < before:
            logger.info("[A] REMOVE_STREAM 成功：%s（剩余 %d 路）", u, after)
        else:
            logger.info("[A] REMOVE_STREAM 未找到：%s（保持 %d 路）", u, after)

    def _handle_update_stream(msg: Dict[str, Any], cur_list: List[Any]) -> None:
        """
        动态更新一条流（原位替换）：
        兼容 {type:'RTSP_UPDATE_STREAM'|'UPDATE_STREAM', old_rtsp_url: str, item: {..}, index?: int}
        - SECURITY_SINGLE：只替换索引 0（且最好匹配 old_rtsp_url）
        - SECURITY_POLLING：优先使用 index；否则用 old_rtsp_url 定位；防重复
        """
        old_url = msg.get("old_rtsp_url")
        item = msg.get("item") or {}
        idx_hint = msg.get("index")

        # 构造新 RTSP
        try:
            if isinstance(item, RTSP):
                new_obj = item
            elif isinstance(item, dict):
                new_obj = RTSP(**item)
            else:
                logger.warning("[A] UPDATE_STREAM item 类型无效：%r", type(item))
                return
        except Exception as e:
            logger.warning("[A] UPDATE_STREAM 构造 RTSP 失败：%s", e)
            return

        if cur_mode == MODEL.SECURITY_SINGLE:
            if not cur_list:
                logger.warning("[A] UPDATE_STREAM：单流模式但列表为空")
                return
            # 避免误改：如果给了 old_url 且不匹配，提示并继续按 0 替
            if old_url and cur_list[0].rtsp_url != old_url:
                logger.warning(
                    "[A] UPDATE_STREAM(SECURITY_SINGLE) old_rtsp_url 不匹配：%s != %s（仍将替换索引0）",
                    cur_list[0].rtsp_url,
                    old_url,
                )
            cur_list[0] = new_obj
            logger.info("[A] 单流已更新：%s → %s", old_url, new_obj.rtsp_url)
            return

        if cur_mode == MODEL.SECURITY_POLLING:
            if not cur_list:
                logger.warning("[A] UPDATE_STREAM：轮询模式但列表为空")
                return

            # 定位索引：优先 index，其次 old_url
            repl_idx = None
            if isinstance(idx_hint, int) and 0 <= idx_hint < len(cur_list):
                repl_idx = idx_hint
            elif old_url:
                for i, it in enumerate(cur_list):
                    if getattr(it, "rtsp_url", None) == old_url:
                        repl_idx = i
                        break
            if repl_idx is None:
                logger.warning("[A] UPDATE_STREAM 未找到可替换项（old_rtsp_url/index 无效）")
                return

            # 防止与其他路重复
            for j, it in enumerate(cur_list):
                if j == repl_idx:
                    continue
                if getattr(it, "rtsp_url", None) == new_obj.rtsp_url:
                    logger.warning("[A] UPDATE_STREAM 新 URL 与其他路重复：%s（放弃）", new_obj.rtsp_url)
                    return

            cur_list[repl_idx] = new_obj
            logger.info("[A] 轮询路[%d] 已更新：%s → %s", repl_idx, old_url, new_obj.rtsp_url)
            return

        logger.warning("[A] UPDATE_STREAM：当前模式不支持：%s", cur_mode)

    def _handle_update_interval(msg: Dict[str, Any], batch_obj: Optional[RTSPBatchConfig]) -> None:
        """
        动态更新轮询间隔：兼容 {type:'RTSP_UPDATE_INTERVAL'|'UPDATE_INTERVAL', polling_batch_interval:int}
        仅 SECURITY_POLLING 下有效。
        """
        if cur_mode != MODEL.SECURITY_POLLING:
            logger.warning("[A] UPDATE_INTERVAL 仅在 SECURITY_POLLING 下有效（当前：%s）", cur_mode)
            return
        if not batch_obj:
            logger.warning("[A] UPDATE_INTERVAL 收到但 batch_obj 未初始化")
            return
        try:
            ni = float(msg.get("polling_batch_interval"))
        except Exception:
            logger.warning("[A] UPDATE_INTERVAL 值无效：%r", msg.get("polling_batch_interval"))
            return
        if ni < 10.0:
            logger.warning("[A] UPDATE_INTERVAL 过小，被拒：%.3fs < 10.0", ni)
            return
        old = batch_obj.polling_batch_interval
        batch_obj.polling_batch_interval = ni
        logger.info(f"[A] 轮询间隔已更新: {old} → {ni}（下一轮生效）")

    def _handle_update_cut_window_sec(msg: Dict[str, Any]) -> None:
        nonlocal cut_window_sec, cur_mode

        if cur_mode not in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING):
            logger.warning(
                "[A] UPDATE_CUT_WINDOW_SEC 仅在 SECURITY_SINGLE / SECURITY_POLLING 下有效（当前：%s）",
                cur_mode,
            )
            return

        raw = msg.get("cut_window_sec")
        try:
            cws = float(raw)
        except (TypeError, ValueError):
            logger.warning("[A] UPDATE_CUT_WINDOW_SEC 值无效：%r", raw)
            return

        if cws < 1.0:
            logger.warning("[A] UPDATE_CUT_WINDOW_SEC 过小, 被拒: %.3f < 1.0", cws)
            return

        old = cut_window_sec
        cut_window_sec = cws
        logger.info("[A] 切片时长已更新: %.3fs → %.3fs (下一流生效)", old, cws)

    def drain_ctrl_once(
        current_streams: Optional[List[Any]] = None,
        batch_obj: Optional[RTSPBatchConfig] = None
    ) -> Optional[str]:
        nonlocal running, paused, cur_mode, resume_jump_to_live
        try:
            msg = q_ctrl.get_nowait()
        except Empty:
            return None

        if msg is stop:
            return "STOP"

        if isinstance(msg, dict):
            typ = msg.get("type")
            if typ == "START":
                running = True
                logger.info("[A] START")
            elif typ == "PAUSE":
                paused = True
                logger.info("[A] PAUSE")
            elif typ == "RESUME":
                paused = False
                # RTSP 模式：恢复后对齐到直播点；OFFLINE 不跳
                if cur_mode in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING):
                    resume_jump_to_live = True
                    # 清理下游未消费的“旧切片”
                    _drain_queue_completely(q_video)
                    _drain_queue_completely(q_audio)
                    logger.info("[A] RESUME（RTSP）：将跳到直播点，并清理未消费队列。")
                else:
                    logger.info("[A] RESUME（OFFLINE）：继续旧 t0。")
            elif typ == "MODE_CHANGE":
                val = msg.get("value")
                try:
                    cur_mode_local = val if isinstance(val, MODEL) else MODEL(str(val))
                    cur_mode = cur_mode_local
                    logger.info(f"[A] MODE_CHANGE → {cur_mode}")
                except Exception:
                    logger.warning("[A] MODE_CHANGE 值无法解析：%r", val)

            # --- 动态流：兼容 RTSP_* 与无前缀 ---
            elif typ in ("ADD_STREAM", "RTSP_ADD_STREAM") and current_streams is not None:
                _handle_add_stream(msg, current_streams)
            elif typ in ("REMOVE_STREAM", "RTSP_REMOVE_STREAM") and current_streams is not None:
                _handle_remove_stream(msg, current_streams)
            elif typ in ("RTSP_UPDATE_STREAM", "UPDATE_STREAM") and current_streams is not None:
                _handle_update_stream(msg, current_streams)
            elif typ in ("RTSP_UPDATE_INTERVAL", "UPDATE_INTERVAL"):
                _handle_update_interval(msg, batch_obj)
            elif typ in ("RTSP_UPDATE_CUT_WINDOW_SEC", "UPDATE_CUT_WINDOW_SEC"):
                _handle_update_cut_window_sec(msg)

            elif typ in ("STOP", "SHUTDOWN"):
                return "STOP"
        return None

    def _wait_run(
        current_streams: Optional[List[Any]] = None,
        batch_obj: Optional[RTSPBatchConfig] = None
    ):
        """等待 running 且非 paused；期间处理控制与动态流。"""
        while True:
            res = drain_ctrl_once(current_streams, batch_obj)
            if res == "STOP":
                return False
            if running and (not paused):
                return True
            sleep(0.02)

    # ---- 公共一次切窗 + 关键帧 + 投递 ----
    def _cut_emit_once(
        *,
        src_url: str,
        segment_index: int,
        tag_prefix: str,
        have_audio: bool,
        stream_index: Optional[int] = None,
        stream_seg_index: Optional[int] = None,
        polling_round_index: Optional[int] = None,
        rtsp_system_prompt: Optional[object] = None,
        step_hint_t0: float = 0.0,
        window_index_in_stream: int = 0,
        on_result: Optional[Callable[[bool], None]] = None,
    ) -> bool:
        """
        返回 False 表示检测到 STOP 或致命失败需退出上层循环。
        注意：对 RTSP，ffmpeg 实际抓取应当以“当前直播点”为主；step_hint_t0 仅用于时间轴元数据。

        on_result: 可选回调，参数为 bool：
            True  表示本次切片成功（产出了有效音/视频）；
            False 表示本次切片失败（ffmpeg 异常 / 结果为空），用于轮询模式统计坏流。
        """
        try:
            seg = cut_and_standardize_segment(
                src_url=src_url,
                start_time=step_hint_t0,
                duration=cut_window_sec,
                output_dir=OUT_DIR,
                segment_index=segment_index,
                have_audio=have_audio,
            )
        except Exception as e:
            logger.error(
                "[A] 切片失败(src=%s, seg_index=%d): %s, 跳过该切片",
                src_url,
                segment_index,
                e,
            )
            if on_result is not None:
                try:
                    on_result(False)
                except Exception as cb_e:
                    logger.warning("[A] on_result 回调异常(失败路径)：%s", cb_e)
            # 对 RTSP/轮询模式：单路/单片失败直接跳过；OFFLINE 也继续下一片段
            return True

        v_out = seg.get("video_path")
        a_out = seg.get("audio_path")
        has_v = _file_exists_nonzero(v_out)
        has_a = _file_exists_nonzero(a_out)

        # 如果既没有视频也没有音频，视为一次失败
        if not has_v and not has_a:
            logger.warning(
                "[A] seg#%s 切片结果为空(src=%s)，视为失败，跳过该片段。",
                segment_index,
                src_url,
            )
            if on_result is not None:
                try:
                    on_result(False)
                except Exception as cb_e:
                    logger.warning("[A] on_result 回调异常(空结果路径)：%s", cb_e)
            return True

        # 时间元数据
        seg_t0_rel = float(seg.get("t0") or 0.0)
        seg_t1_rel = float(seg.get("t1") or (seg_t0_rel + cut_window_sec))
        seg_t0_epoch = seg.get("t0_epoch")
        seg_t1_epoch = seg.get("t1_epoch")
        seg_t0_iso = seg.get("t0_iso") or (_epoch_to_iso_utc(seg_t0_epoch) if seg_t0_epoch else None)
        seg_t1_iso = seg.get("t1_iso") or (_epoch_to_iso_utc(seg_t1_epoch) if seg_t1_epoch else None)

        # 关键帧挑选 & 投递
        if has_v and q_video is not None:
            tag = f"{tag_prefix}_seg{segment_index:06d}"
            paths, pts, idxs = pick_best_change_frame_with_pts(
                v_out,
                OUT_DIR,
                tag,
                seg_t0=seg_t0_rel,
                alpha_bgr=alpha_bgr,
                topk_frames=topk_frames,
            )
            if not paths:
                paths, pts, idxs = sample_one_frame_with_pts(
                    v_out,
                    OUT_DIR,
                    tag,
                    seg_t0=seg_t0_rel,
                )

            if paths:
                if seg_t0_epoch is not None:
                    frame_epoch = [float(seg_t0_epoch) + (float(p) - seg_t0_rel) for p in pts]
                    frame_iso = [_epoch_to_iso_utc(ep) for ep in frame_epoch]
                else:
                    frame_epoch = []
                    frame_iso = []

                payload_video: Dict[str, object] = {
                    "path": v_out,
                    "t0": seg_t0_rel,
                    "t1": seg_t1_rel,
                    "t0_iso": seg_t0_iso,
                    "t1_iso": seg_t1_iso,
                    "t0_epoch": seg_t0_epoch,
                    "t1_epoch": seg_t1_epoch,
                    "clip_t0": seg_t0_rel,
                    "clip_t1": seg_t1_rel,
                    "segment_index": segment_index,
                    "keyframes": paths,  # 关键帧存放路径
                    "frame_pts": pts,
                    "frame_indices": idxs,
                    "frame_epoch": frame_epoch,
                    "frame_iso": frame_iso,
                    "small_video": None,
                    # 透传源流信息（上游/前端渲染）
                    "stream_url": src_url,
                    "stream_index": stream_index,
                    "stream_segment_index": stream_seg_index,
                    "window_index_in_stream": window_index_in_stream,
                    "polling_round_index": polling_round_index,
                    "rtsp_system_prompt": rtsp_system_prompt,  # 可以是枚举或 str
                }
                if not _safe_put_with_ctrl(q_video, payload_video, q_ctrl, stop):
                    return False
            else:
                logger.warning("[A] seg#%s 无可用证据帧，跳过视频侧。", segment_index)

        # 音频侧
        if has_a and have_audio and q_audio is not None:
            audio_payload = {
                "path": a_out,
                "t0": seg_t0_rel,
                "t1": seg_t1_rel,
                "t0_iso": seg_t0_iso,
                "t1_iso": seg_t1_iso,
                "t0_epoch": seg_t0_epoch,
                "t1_epoch": seg_t1_epoch,
                "segment_index": segment_index,
                "stream_index": stream_index,
                "stream_segment_index": stream_seg_index,
                "stream_url": src_url,
                "polling_round_index": polling_round_index,
            }
            if not _safe_put_with_ctrl(q_audio, audio_payload, q_ctrl, stop):
                return False

        # 至少有一条音/视频产出，视为成功
        if on_result is not None:
            try:
                on_result(True)
            except Exception as cb_e:
                logger.warning("[A] on_result 回调异常(成功路径)：%s", cb_e)

        return True

    # ===================== 三种模式的工作循环 =====================
    def _loop_offline(local_url: str):
        """OFFLINE：恢复后继续旧 t0，不清队列。"""
        seg_idx = 0
        t0 = 0.0
        max_duration = _probe_duration(local_url)
        tail_emitted = False

        logger.info("[A] OFFLINE 循环：duration=%s", max_duration)
        while True:
            # 控制优先
            for _ in range(4):
                res = drain_ctrl_once()
                if res == "STOP":
                    return
                if not running or paused:
                    sleep(0.01)
            if not running or paused:
                continue

            # EOF
            if max_duration is not None and t0 >= max_duration:
                if not tail_emitted and max_duration > 0:
                    new_t0 = max(0.0, max_duration - cut_window_sec)
                    if new_t0 + 1e-6 < t0:
                        t0 = new_t0
                        tail_emitted = True
                    else:
                        logger.info("[A] 离线完成，退出。")
                        return
                else:
                    logger.info("[A] 离线完成，退出。")
                    return

            ok = _cut_emit_once(
                src_url=local_url,
                segment_index=seg_idx,
                tag_prefix="offline",
                have_audio=have_audio_track,
                step_hint_t0=t0,
            )
            if not ok:
                return

            seg_idx += 1
            t0 += cut_window_sec
            if tail_emitted and (max_duration is not None):
                t0 = max_duration
            sleep(0.005)

    def _loop_single_rtsp(rtsp_url: str, rtsp_prompt_obj: Optional[object]):
        """单流 RTSP：恢复后跳直播点，并清空未消费旧片段。"""
        nonlocal resume_jump_to_live

        global_seg_idx = 0
        t0_rel = 0.0
        logger.info("[A] SECURITY_SINGLE 循环：%s", rtsp_url)

        while True:
            # 控制优先
            for _ in range(4):
                res = drain_ctrl_once()
                if res == "STOP":
                    return
                if not running or paused:
                    sleep(0.01)
            if not running or paused:
                continue

            # 恢复 → 跳直播点 + 清队列（前面已做；这里只需重置相对轴并清标志）
            if resume_jump_to_live:
                t0_rel = 0.0
                resume_jump_to_live = False
                logger.info("[A] 单流：已对齐直播点（相对轴清零）。")

            ok = _cut_emit_once(
                src_url=rtsp_url,
                segment_index=global_seg_idx,
                tag_prefix="rtsp_s00",
                have_audio=have_audio_track,
                stream_index=0,
                stream_seg_index=global_seg_idx,  # 单流时：全局 seg 等价“该流的 seg”
                polling_round_index=None,
                rtsp_system_prompt=rtsp_prompt_obj,
                step_hint_t0=t0_rel,
            )
            if not ok:
                return

            global_seg_idx += 1
            t0_rel += cut_window_sec
            sleep(0.003)

    def _loop_polling_rtsp(initial_batch: RTSPBatchConfig):
        """
        轮询 RTSP：
        - 支持 ADD/REMOVE/UPDATE_STREAM、UPDATE_INTERVAL 控制；
        - 恢复后跳直播点（清空未消费队列），并把每路相对轴清零；
        - 新增：坏流检测与拉黑机制，仅在 SECURITY_POLLING 模式生效：
            * 某一路连续 BAD_FAIL_THRESHOLD 次切片失败 → 标记为坏流（拉黑）；
            * N 轮后尝试切片 1 次恢复；
            * 若恢复失败 → 按轮次指数退避，尝试间隔轮数 <= min(16, 2N)；
            * 恢复成功 → 清空坏流标记与计数，恢复正常轮询。
        """
        nonlocal resume_jump_to_live

        # 当前活跃流列表（可被动态增删/改）
        current_streams: List[Any] = list(initial_batch.polling_list or [])
        # 每路分段号、相对轴
        stream_seg_index: Dict[int, int] = {i: 0 for i in range(len(current_streams))}
        stream_t0_rel: Dict[int, float] = {i: 0.0 for i in range(len(current_streams))}
        # 全局分段号 & 轮询轮数
        global_seg_idx = 0
        polling_round_index = 0

        # 坏流相关参数
        BAD_FAIL_THRESHOLD = 3  # N：连续 N 次失败视为坏流，可按需调整
        MAX_RETRY_GAP_ROUNDS = 16  # 尝试间隔轮数硬上限
        # cap_gap = min(16, 2N)
        RETRY_GAP_CAP = min(MAX_RETRY_GAP_ROUNDS, 2 * BAD_FAIL_THRESHOLD)

        # 每路连续失败计数（仅正常阶段使用，坏流恢复阶段不再累加）
        stream_fail_count: Dict[int, int] = {i: 0 for i in range(len(current_streams))}
        # 每路是否被标记为坏流
        bad_stream_flag: Dict[int, bool] = {i: False for i in range(len(current_streams))}
        # 每路坏流：下一次尝试恢复的“轮次编号”
        bad_stream_next_round: Dict[int, int] = {}
        # 每路坏流：当前尝试间隔轮数（会指数退避，但不超过 RETRY_GAP_CAP）
        bad_stream_retry_gap: Dict[int, int] = {}

        logger.info(
            "[A] SECURITY_POLLING 循环：初始 %d 路, batch_interval=%ss, "
            "坏流阈值=%d, 最大尝试间隔轮=%d, cap_gap=%d",
            len(current_streams),
            int(initial_batch.polling_batch_interval),
            BAD_FAIL_THRESHOLD,
            MAX_RETRY_GAP_ROUNDS,
            RETRY_GAP_CAP,
        )

        # 辅助：当 current_streams 变化后，重建映射字典（保持已存在流的状态不丢）
        def _rebuild_per_stream_maps():
            nonlocal stream_seg_index, stream_t0_rel
            nonlocal stream_fail_count, bad_stream_flag, bad_stream_next_round, bad_stream_retry_gap

            new_seg_index: Dict[int, int] = {}
            new_t0_rel: Dict[int, float] = {}
            new_fail_count: Dict[int, int] = {}
            new_bad_flag: Dict[int, bool] = {}
            new_next_round: Dict[int, int] = {}
            new_retry_gap: Dict[int, int] = {}

            for i, _item in enumerate(current_streams):
                # seg & t0
                if i in stream_seg_index:
                    new_seg_index[i] = stream_seg_index[i]
                    new_t0_rel[i] = stream_t0_rel[i]
                else:
                    new_seg_index[i] = 0
                    new_t0_rel[i] = 0.0

                # 连续失败计数
                if i in stream_fail_count:
                    new_fail_count[i] = stream_fail_count[i]
                else:
                    new_fail_count[i] = 0

                # 坏流标记
                if i in bad_stream_flag:
                    new_bad_flag[i] = bad_stream_flag[i]
                else:
                    new_bad_flag[i] = False

                # 恢复轮次
                if i in bad_stream_next_round:
                    new_next_round[i] = bad_stream_next_round[i]

                # 间隔配置
                if i in bad_stream_retry_gap:
                    new_retry_gap[i] = bad_stream_retry_gap[i]

            stream_seg_index = new_seg_index
            stream_t0_rel = new_t0_rel
            stream_fail_count = new_fail_count
            bad_stream_flag = new_bad_flag
            bad_stream_next_round = new_next_round
            bad_stream_retry_gap = new_retry_gap

        def _make_on_result(si: int, round_idx: int) -> Callable[[bool], None]:
            """
            为某个流 si 生成一次切片结果回调：
            - success=True  表示该次切片成功；
            - success=False 表示该次切片失败。
            """
            def _on_result(success: bool, si: int = si, round_idx: int = round_idx) -> None:
                nonlocal stream_fail_count, bad_stream_flag, bad_stream_next_round, bad_stream_retry_gap

                if si not in stream_fail_count:
                    stream_fail_count[si] = 0
                if si not in bad_stream_flag:
                    bad_stream_flag[si] = False

                # 正常阶段：还不是坏流
                if not bad_stream_flag[si]:
                    if success:
                        # 任意一次成功清零连续失败计数
                        if stream_fail_count[si]:
                            stream_fail_count[si] = 0
                    else:
                        stream_fail_count[si] += 1
                        if stream_fail_count[si] >= BAD_FAIL_THRESHOLD:
                            # 连续 N 次失败 → 标记为坏流
                            bad_stream_flag[si] = True
                            stream_fail_count[si] = 0

                            # 首次恢复尝试：N 轮后
                            first_gap = BAD_FAIL_THRESHOLD
                            gap = min(first_gap, RETRY_GAP_CAP)
                            bad_stream_retry_gap[si] = gap
                            next_round = round_idx + gap
                            bad_stream_next_round[si] = next_round
                            logger.warning(
                                "[A] 轮询流[%d] 连续 %d 次切片失败，标记为坏流，将在 %d 轮后尝试恢复（目标轮=%d，cap_gap=%d）。",
                                si,
                                BAD_FAIL_THRESHOLD,
                                gap,
                                next_round,
                                RETRY_GAP_CAP,
                            )
                    return

                # 坏流恢复尝试阶段
                if success:
                    # 恢复成功：取消坏流标记 & 清空状态
                    bad_stream_flag[si] = False
                    stream_fail_count[si] = 0
                    bad_stream_retry_gap[si] = BAD_FAIL_THRESHOLD
                    if si in bad_stream_next_round:
                        bad_stream_next_round.pop(si, None)
                    logger.info("[A] 轮询流[%d] 恢复成功，取消坏流标记。", si)
                else:
                    # 恢复失败：指数退避，但尝试间隔轮数不超过 min(16, 2N)
                    prev_gap = bad_stream_retry_gap.get(si, BAD_FAIL_THRESHOLD)
                    new_gap = max(prev_gap * 2, BAD_FAIL_THRESHOLD)
                    new_gap = min(new_gap, RETRY_GAP_CAP)
                    bad_stream_retry_gap[si] = new_gap
                    next_round = round_idx + new_gap
                    bad_stream_next_round[si] = next_round
                    logger.warning(
                        "[A] 轮询流[%d] 恢复尝试失败，将在 %d 轮后再次尝试（目标轮=%d，间隔不超过 %d 轮）。",
                        si,
                        new_gap,
                        next_round,
                        RETRY_GAP_CAP,
                    )

            return _on_result

        while True:
            # 防止无流
            if not current_streams:
                # 等待控制消息把流加回来
                logger.info("[A] SECURITY_POLLING 当前无流，等待控制指令 ADD_STREAM ...")
                if not _wait_run(current_streams, initial_batch):
                    return
                if not current_streams:
                    sleep(0.2)
                    continue

            # —— 一轮开始（允许在轮内动态指令；索引以本轮快照为准）——
            round_streams_snapshot = list(current_streams)  # 快照：本轮固定集合
            for si, item in enumerate(round_streams_snapshot):
                # 控制优先（收控制 / 动态指令）
                for _ in range(4):
                    res = drain_ctrl_once(current_streams, initial_batch)
                    if res == "STOP":
                        return
                    if not running or paused:
                        sleep(0.01)
                if not running or paused:
                    continue

                # 恢复 → 跳直播点：清标志、每路相对轴清零、队列已清
                if resume_jump_to_live:
                    for k in stream_t0_rel.keys():
                        stream_t0_rel[k] = 0.0
                    resume_jump_to_live = False
                    logger.info("[A] 轮询：已对齐直播点（各路相对轴清零）。")

                # 坏流处理：若标记为坏流，则按轮次决定“跳过”还是“尝试恢复（切 1 次）”
                if bad_stream_flag.get(si):
                    nr = bad_stream_next_round.get(si)
                    if nr is None:
                        # 理论上不应出现；兜底：按当前轮后 BAD_FAIL_THRESHOLD 轮再试
                        gap = bad_stream_retry_gap.get(si, BAD_FAIL_THRESHOLD)
                        gap = min(max(gap, BAD_FAIL_THRESHOLD), RETRY_GAP_CAP)
                        bad_stream_retry_gap[si] = gap
                        nr = polling_round_index + gap
                        bad_stream_next_round[si] = nr

                    if polling_round_index < nr:
                        # 尚未到恢复尝试轮：完全跳过该流本轮的切片
                        logger.debug(
                            "[A] 轮询流[%d] 为坏流状态，目标恢复轮=%d，本轮(%d)跳过。",
                            si,
                            nr,
                            polling_round_index,
                        )
                        continue
                    else:
                        # 到达恢复轮次：只切 1 个窗口进行探测
                        try:
                            rtsp_url = item.rtsp_url
                        except Exception:
                            logger.warning("[A] 轮询恢复：流[%d] 无效条目，跳过。", si)
                            continue

                        rtsp_prompt_obj = getattr(item, "rtsp_system_prompt", None) \
                            or getattr(item, "rtsp_prompt", None)

                        on_result = _make_on_result(si, polling_round_index)
                        ok = _cut_emit_once(
                            src_url=rtsp_url,
                            segment_index=global_seg_idx,
                            tag_prefix=f"rtsp_s{si:02d}",
                            have_audio=have_audio_track,
                            stream_index=si,                                # 使用“本轮快照索引”对齐前端
                            stream_seg_index=stream_seg_index.get(si, 0),   # 每路累计 seg
                            polling_round_index=polling_round_index,
                            rtsp_system_prompt=rtsp_prompt_obj,
                            step_hint_t0=stream_t0_rel.get(si, 0.0),
                            window_index_in_stream=0,
                            on_result=on_result,
                        )
                        if not ok:
                            return

                        global_seg_idx += 1
                        stream_seg_index[si] = stream_seg_index.get(si, 0) + 1
                        stream_t0_rel[si] = stream_t0_rel.get(si, 0.0) + cut_window_sec
                        sleep(0.003)
                        # 本轮内不再对这路多切，继续下一个流
                        continue

                # 正常流：按配置的 cuts 进行切片，并统计失败次数
                try:
                    rtsp_url = item.rtsp_url
                except Exception:
                    logger.warning("[A] 轮询：遇到非法 RTSP 条目，跳过。")
                    continue

                cuts = 1
                try:
                    cuts = max(1, int(getattr(item, "rtsp_cut_number", 1)))
                except Exception:
                    pass

                rtsp_prompt_obj = getattr(item, "rtsp_system_prompt", None) \
                    or getattr(item, "rtsp_prompt", None)

                for k in range(cuts):
                    on_result = _make_on_result(si, polling_round_index)
                    ok = _cut_emit_once(
                        src_url=rtsp_url,
                        segment_index=global_seg_idx,
                        tag_prefix=f"rtsp_s{si:02d}",
                        have_audio=have_audio_track,
                        stream_index=si,                                # 使用“本轮快照索引”对齐前端
                        stream_seg_index=stream_seg_index.get(si, 0),   # 每路累计 seg
                        polling_round_index=polling_round_index,
                        rtsp_system_prompt=rtsp_prompt_obj,
                        step_hint_t0=stream_t0_rel.get(si, 0.0),
                        window_index_in_stream=k,
                        on_result=on_result,
                    )
                    if not ok:
                        return

                    global_seg_idx += 1
                    stream_seg_index[si] = stream_seg_index.get(si, 0) + 1
                    stream_t0_rel[si] = stream_t0_rel.get(si, 0.0) + cut_window_sec
                    sleep(0.003)

            # ——一轮结束：应用“轮内”动态变更，重建映射，并进入轮询间隔——
            _rebuild_per_stream_maps()
            polling_round_index += 1

            # 轮询间隔：A 侧再做一次兜底，保证 >=10s
            try:
                interval = float(initial_batch.polling_batch_interval)
            except Exception:
                interval = 10.0

            if interval < 10.0:
                logger.warning(
                    "[A] 轮询间隔过小(%.3fs)，按 10s 兜底。",
                    interval,
                )
                interval = 10.0
                try:
                    initial_batch.polling_batch_interval = interval
                except Exception:
                    pass

            if interval > 0:
                wait_until = now_ts() + interval

                now_date_formatted, wait_date_formatted = get_now_wait_formatted_time(wait_until)
                logger.info(
                    f"[A] {now_date_formatted} 本轮轮询结束, 进入间隔等待 {interval}s, "
                    f"预计将在 {wait_date_formatted} 开始下一轮循环"
                )
                while True:
                    # 先处理各种控制指令（STOP / START / PAUSE / RESUME / 动态改流）
                    res = drain_ctrl_once(current_streams, initial_batch)
                    if res == "STOP":
                        logger.info("[A] 收到 STOP 退出")
                        return

                    # 如果处于未 start 或 PAUSE 状态，这里直接挂起，直到恢复
                    if (not running) or paused:
                        logger.info("[A] 间隔等待中收到 PAUSE/未启动，挂起直到 RESUME/START ...")
                        # 等到 running 且不再 paused
                        if not _wait_run(current_streams, initial_batch):
                            # _wait_run 返回 False 说明收到 STOP
                            logger.info("[A] _wait_run 返回 STOP，退出间隔等待")
                            return

                        # 从 PAUSE/未启动 状态恢复以后，把间隔计时重新开始算
                        wait_until = now_ts() + interval
                        now_date_formatted, wait_date_formatted = get_now_wait_formatted_time(wait_until)
                        logger.info(
                            f"[A] {now_date_formatted} 从 PAUSE/未启动 恢复，间隔重新计时 {interval}s, "
                            f"预计将在 {wait_date_formatted} 开始下一轮循环"
                        )
                        continue

                    # 已经是 running 且非 paused，检查是否等够时间
                    if now_ts() >= wait_until:
                        break

                    sleep(0.1)

    # ===================== 主流程入口 =====================
    try:
        # 等待 START
        if not _wait_run():
            return

        if cur_mode == MODEL.OFFLINE:
            if not url:
                logger.error("[A] OFFLINE 模式需要本地文件 url。")
                return
            _loop_offline(url)

        elif cur_mode == MODEL.SECURITY_SINGLE:
            if not (
                rtsp_batch_config
                and rtsp_batch_config.polling_list
                and rtsp_batch_config.polling_list[0].rtsp_url
            ):
                logger.error("[A] SECURITY_SINGLE 需要 rtsp_batch_config.polling_list[0].rtsp_url")
                return
            first = rtsp_batch_config.polling_list[0]
            prompt_obj = getattr(first, "rtsp_system_prompt", None) or getattr(first, "rtsp_prompt", None)
            _loop_single_rtsp(first.rtsp_url, prompt_obj)

        elif cur_mode == MODEL.SECURITY_POLLING:
            if not (
                rtsp_batch_config
                and rtsp_batch_config.polling_list
                and len(rtsp_batch_config.polling_list) >= 2
            ):
                logger.error("[A] SECURITY_POLLING 需要至少 2 条流")
                return
            _loop_polling_rtsp(rtsp_batch_config)

        else:
            logger.error("[A] 未知模式：%s", cur_mode)

    except Exception as e:
        logger.error(f"[A] 运行异常：{e}")
    finally:
        logger.info("[A] 线程退出清理完成")
