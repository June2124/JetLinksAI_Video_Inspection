# -*- coding: utf-8 -*-
from __future__ import annotations

"""
子线程A：切窗 + 标准化 + （按模式）关键帧策略（含 OpenCV/FFmpeg 双兜底）
- 固定参数：url / have_audio_track / mode / slice_sec（启动时传入）
- 运行期控制：q_ctrl 收到 {type: ...}
    START / PAUSE / RESUME / MODE_CHANGE / UPDATE_SLICE / UPDATE_OVERLAP / STOP
- 输出（发往下游 B/C 的“数据”队列，A 不负责下游 STOP 哨兵）：
    (若启用B且存在视频)
    q_video.put({
        "path": <无声小视频>, "t0": <秒>, "t1": <秒>,
        "clip_t0": <秒>, "clip_t1": <秒>,
        "segment_index": <int>,
        "mode": "offline|online|security",
        "win": <float>, "step": <float>, "overlap_sec": <float>,
        "has_audio": <bool>,
        "keyframes": [<jpg路径>...],            # 关键帧/快照（若策略选择了图像）
        "frame_pts": [<float秒>...],            # 与 keyframes 一一对应（源时间线）
        "frame_indices": [<int>...],            # 可选：帧索引（相对该片段 v_out）
        "keyframe_count": <int>,
        "small_video": <压缩后mp4路径或 None>,
        "small_video_fps": <float或None>,       # 若走小视频策略则提供
        "hires": <bool>,
        "policy": {
            "significant_motion": <bool>,
            "policy_used": "small_video" | "fixed_sampling" | "keyframes" | "none",
            "interval_sec": <float>,
            "max_frames": <int|None>,
            "silence_hint": {"silence_ratio": <float>, "is_mostly_silent": <bool>}
        }
    })
    (若启用C且存在音频)
    q_audio.put({
        "path": <标准化wav>, "t0": <秒>, "t1": <秒>,
        "segment_index": <int>,
        "silence_hint": {"silence_ratio": <float>, "is_mostly_silent": <bool>}
    })
"""

from typing import Optional, Tuple, List, Dict
from queue import Queue, Empty, Full
from time import sleep
import subprocess
import numpy as np
import cv2
import time
import os

from src.all_enum import MODEL
from src.utils.ffmpeg_utils import FFmpegUtils
from src.utils.logger_utils import get_logger

logger = get_logger(__name__)


# -------------------- 小工具 --------------------
def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def _imwrite_jpg(path: str, img, quality: int = 90) -> str:
    """imencode 写盘，避免中文路径问题。"""
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)])
    if not ok:
        raise RuntimeError("cv2.imencode('.jpg', ...) failed")
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        f.write(buf.tobytes())
    return path


def _get_video_meta(video_path: str) -> Tuple[float, int]:
    """返回 (fps, total_frames)；失败时回退 (25.0, 0)"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return 25.0, 0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0:
        fps = 25.0
    return fps, total


def _file_exists_nonzero(path: Optional[str]) -> bool:
    return bool(path) and os.path.exists(path) and os.path.getsize(path) > 0


# -------------------- 窗口/步长策略 --------------------
def _mode_window_policy(mode: MODEL, desire_win: float, overlap_sec: float) -> Tuple[float, float]:
    """
    返回 (win, step)
    - security: win ∈ [4, 8],   step = win
    - online  : win ∈ [5, 10],  step = win
    - offline : win ∈ [8, 15],  step = win - overlap_sec (overlap ∈ [0.0, 1.0])
    """
    if mode == MODEL.SECURITY:
        win = _clamp(float(desire_win), 4.0, 12.0)  # 建议 4~8
        return win, win
    elif mode == MODEL.ONLINE:
        win = _clamp(float(desire_win), 5.0, 16.0)  # 建议 5~10
        return win, win
    else:
        win = _clamp(float(desire_win), 8.0, 30.0)  # 建议 8~15
        ovl = _clamp(float(overlap_sec), 0.0, 2.0)
        step = max(0.1, win - ovl)
        return win, step


# -------------------- 运动评分（探测，不写盘） --------------------
def compute_motion_score(
    video_path: str,
    sample_interval: float = 1.0,
    resize_w: int = 160,
    max_samples: int = 30,
) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"avg_diff": 0.0, "max_diff": 0.0, "samples": 0}

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    step = max(1, int(round(fps * sample_interval)))
    last_gray = None
    diffs: List[float] = []
    try:
        i = 0
        while len(diffs) < max_samples:
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            if i % step:
                continue
            h, w = frame.shape[:2]
            small_h = max(1, int(h * (resize_w / max(1, w))))
            small = cv2.resize(frame, (resize_w, small_h), interpolation=cv2.INTER_AREA)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            if last_gray is not None:
                diffs.append(cv2.absdiff(last_gray, gray).mean())
            last_gray = gray
    finally:
        cap.release()

    if not diffs:
        return {"avg_diff": 0.0, "max_diff": 0.0, "samples": 0}
    return {"avg_diff": float(sum(diffs) / len(diffs)), "max_diff": float(max(diffs)), "samples": len(diffs)}


def fast_motion_detect_placeholder(
    video_path: str,
    sample_interval: float = 1.0,
    diff_threshold: float = 15.0,
    max_samples: int = 30,
) -> bool:
    score = compute_motion_score(video_path, sample_interval, 160, max_samples)
    return score["samples"] > 0 and score["avg_diff"] >= diff_threshold


# -------------------- 静音探测（基于标准化后的音频 a_out） --------------------
def silence_detect_hint(
    audio_path: str,
    *,
    noise_db: float = -40.0,
    min_silence: float = 0.30,
    timeout_sec: float = 60.0,
) -> dict:
    """
    使用 FFmpeg silencedetect 对标准化后的 WAV（16kHz/mono/PCM16）做静音探测。
    返回：{"silence_ratio": <0~1>, "is_mostly_silent": <bool>, "segments": [(s,e)...]}
    """
    import re, subprocess
    try:
        duration = FFmpegUtils.ffprobe_duration(audio_path)
    except Exception:
        duration = None

    start_re = re.compile(r"silence_start:\s*([0-9.]+)")
    end_re = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-v", "info",
        "-i", audio_path,
        "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
        "-f", "null", "-"
    ]

    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1)
    except FileNotFoundError:
        return {"silence_ratio": 0.0, "is_mostly_silent": False, "segments": []}

    silence_intervals: List[tuple] = []
    cur_start = None
    t0 = time.time()
    try:
        assert proc.stderr
        for line in iter(proc.stderr.readline, ""):
            if timeout_sec and (time.time() - t0 > timeout_sec):
                break
            m1 = start_re.search(line)
            if m1:
                try:
                    cur_start = float(m1.group(1))
                except Exception:
                    cur_start = None
                continue
            m2 = end_re.search(line)
            if m2:
                try:
                    end = float(m2.group(1))
                    dur = float(m2.group(2))
                    silence_intervals.append((cur_start, end, dur))
                except Exception:
                    pass
                cur_start = None
        try:
            proc.terminate()
        except Exception:
            pass
        try:
            proc.wait(timeout=2)
        except Exception:
            pass
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass

    total_silence = sum([d for _, _, d in silence_intervals if d])
    if duration and duration > 0:
        silence_ratio = max(0.0, min(1.0, total_silence / duration))
    else:
        silence_ratio = 0.0

    return {
        "silence_ratio": round(silence_ratio, 3),
        "is_mostly_silent": silence_ratio > 0.8,
        "segments": [(s, e) for (s, e, _) in silence_intervals if s is not None and e is not None],
    }


# -------------------- FFmpeg 抓帧兜底 --------------------
def _ffmpeg_grab_frames_by_indices(video_path: str, out_dir: str, tag: str, indices: List[int]) -> List[str]:
    """
    按近似帧位导出图片（用 eq(n,idx) 选择），部分容器/时间基可能不精确。
    """
    if not indices:
        return []
    os.makedirs(out_dir, exist_ok=True)
    outs: List[str] = []
    for k, idx in enumerate(indices):
        out = os.path.join(out_dir, f"{tag}_snap_{k:04d}.jpg")
        # 注意：某些封装下 eq(n,idx) 并不总能命中；这是兜底手段
        # -vsync vfr 可避免重复帧写出异常
        cmd = [
            "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
            "-i", video_path,
            "-vf", f"select='eq(n\\,{idx})'",
            "-vsync", "vfr",
            out
        ]
        try:
            subprocess.check_call(cmd)
            if _file_exists_nonzero(out):
                outs.append(out)
        except Exception:
            # 如果按索引失败，就跳过这张，整体不报错
            continue
    return outs


def _ffmpeg_grab_frames_by_interval(video_path: str, out_dir: str, tag: str, interval_sec: float) -> List[str]:
    """
    等间隔抓帧（fps=1/interval）。作为 OpenCV 间隔抽帧失败时的兜底。
    """
    if interval_sec <= 0:
        interval_sec = 1.0
    os.makedirs(out_dir, exist_ok=True)
    pattern = os.path.join(out_dir, f"{tag}_kf_%04d.jpg")
    cmd = [
        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-vf", f"fps=1/{max(1e-3, float(interval_sec))}",
        pattern
    ]
    try:
        subprocess.check_call(cmd)
    except Exception:
        return []
    # 收集写出的文件
    outs = []
    for fname in sorted(os.listdir(out_dir)):
        if fname.startswith(f"{tag}_kf_") and fname.endswith(".jpg"):
            outs.append(os.path.join(out_dir, fname))
    return outs


# -------------------- 关键帧/快照（含 PTS） --------------------
def extract_keyframes_by_interval_with_pts(
    video_path: str,
    out_dir: str,
    tag: str,
    interval_sec: float = 1.0,
    diff_threshold: float = 0.65,
    resize_w: int = 320,
    jpg_quality: int = 90,
    max_frames: Optional[int] = None,
    *,
    seg_t0: float,
) -> Tuple[List[str], List[float], List[int]]:
    """
    返回 (paths, pts_list, frame_indices)
    - 优先 OpenCV：按间隔取帧并用相邻差分筛选
    - OpenCV 打不开 → FFmpeg 兜底（等间隔导出）
    """
    # OpenCV 路线
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("[A] 待抽取关键帧视频，无法通过 OpenCV 打开，使用 FFmpeg 兜底")
        # FFmpeg 兜底
        outs = _ffmpeg_grab_frames_by_interval(video_path, out_dir, tag, interval_sec)
        if not outs:
            return [], [], []
        # 没有明确的 pts/索引信息，这里用近似：按照顺序回填
        fps, total = _get_video_meta(video_path)
        if fps <= 0:
            fps = 25.0
        pts = [float(seg_t0) + i * interval_sec for i in range(len(outs))]
        idxs = [min(int(round(i * fps * interval_sec)), max(0, total - 1)) for i in range(len(outs))]
        return outs, pts, idxs

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0
    step = max(1, int(round(fps * interval_sec)))

    paths: List[str] = []
    pts: List[float] = []
    fidx: List[int] = []

    last_small = None
    cur_idx = -1
    try:
        while True:
            # 跳过 step-1 帧，抓一帧参与相邻差分
            for _ in range(step - 1):
                if not cap.grab():
                    # 结束
                    cap.release()
                    return paths, pts, fidx
            ret, frame = cap.read()
            if not ret:
                break
            cur_idx = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1  # 当前帧索引
            h, w = frame.shape[:2]
            small_h = max(1, int(h * (resize_w / max(1, w))))
            small = cv2.resize(frame, (resize_w, small_h), interpolation=cv2.INTER_AREA)

            save = last_small is None
            if last_small is not None:
                diff = cv2.absdiff(last_small, small)
                score = float(np.count_nonzero(diff)) / float(diff.size)
                save = score >= diff_threshold
            last_small = small

            if save:
                name = f"{tag}_kf_{len(paths):04d}.jpg"
                out_path = os.path.join(out_dir, name)
                _imwrite_jpg(out_path, frame, quality=jpg_quality)
                paths.append(out_path)
                pts.append(float(seg_t0) + (cur_idx / fps))
                fidx.append(cur_idx)
                if max_frames and len(paths) >= max_frames:
                    break
    finally:
        cap.release()

    return paths, pts, fidx


def sample_fixed_frames_with_pts(
    video_path: str,
    out_dir: str,
    tag: str,
    count: int = 2,
    jpg_quality: int = 90,
    *,
    seg_t0: float
) -> Tuple[List[str], List[float], List[int]]:
    """
    更稳妥的固定抽帧：优先“顺播”读取，避免 set(CAP_PROP_POS_FRAMES) 的随机 seek 不可靠；
    OpenCV 失败则使用 FFmpeg 按帧索引近似抓帧（或按时间等距抓帧）兜底。
    返回 (paths, pts_list, frame_indices)
    """
    # ---- OpenCV 顺播 ----
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.warning("[A] 待抽取固定帧视频，无法通过 OpenCV 打开，使用 FFmpeg 兜底")
        # 用 meta 估算索引/时间点
        fps, total = _get_video_meta(video_path)
        if total <= 0:
            return [], [], []
        if count <= 1:
            indices = [total // 2]
        else:
            indices = [int(round((k + 1) / (count + 1) * total)) for k in range(count)]
        outs = _ffmpeg_grab_frames_by_indices(video_path, out_dir, tag, indices)
        if not outs:
            # 最后备选：按等间隔
            outs = _ffmpeg_grab_frames_by_interval(video_path, out_dir, tag, max(1.0, (total / max(1, count)) / max(1.0, fps)))
        if not outs:
            return [], [], []
        if fps <= 0:
            fps = 25.0
        pts = [float(seg_t0) + (idx / fps) for idx in range(len(outs))]
        idxs = indices if outs and len(outs) == len(indices) else list(range(len(outs)))
        return outs, pts, idxs

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 25.0)
    if fps <= 0:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        cap.release()
        return [], [], []

    # 目标索引（顺序）
    if count <= 1:
        target_idxs = [total // 2]
    else:
        target_idxs = [int(round((k + 1) / (count + 1) * total)) for k in range(count)]
    target_set = set(target_idxs)
    target_sorted = sorted(target_idxs)

    paths, pts, fidx = [], [], []
    cur_read_idx = -1
    next_target_i = 0

    try:
        while next_target_i < len(target_sorted):
            ret, frame = cap.read()
            if not ret:
                break
            cur_read_idx += 1
            # 直接顺播直到命中目标帧
            if cur_read_idx == target_sorted[next_target_i]:
                name = f"{tag}_snap_{len(paths):04d}.jpg"
                out_path = os.path.join(out_dir, name)
                _imwrite_jpg(out_path, frame, quality=jpg_quality)
                paths.append(out_path)
                pts.append(float(seg_t0) + (cur_read_idx / fps))
                fidx.append(cur_read_idx)
                next_target_i += 1
    finally:
        cap.release()

    if paths:
        return paths, pts, fidx

    # ---- OpenCV 顺播失败 → FFmpeg 兜底 ----
    logger.warning("[A] OpenCV固定抽帧未取到图片，回落到 FFmpeg 抓帧（按索引或等间隔）")
    outs = _ffmpeg_grab_frames_by_indices(video_path, out_dir, tag, target_sorted)
    if not outs:
        # 最后备选：按等间隔
        interval = max(1.0, (total / max(1, count)) / max(1.0, fps))
        outs = _ffmpeg_grab_frames_by_interval(video_path, out_dir, tag, interval)
    if not outs:
        return [], [], []

    # 兜底下的 pts/idx 近似填充
    pts2 = [float(seg_t0) + i * max(1.0 / fps, 0.04) for i in range(len(outs))]
    idxs2 = target_sorted if len(outs) == len(target_sorted) else list(range(len(outs)))
    return outs, pts2, idxs2


# -------------------- 关键帧/小视频策略表 --------------------
VLM_POLICY = {
    MODEL.ONLINE: {
        "interval_sec": 1.0,
        "max_frames": 3,
        "hires": False,
        "small_video": False,
        "small_video_encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"},
    },
    MODEL.SECURITY: {
        "interval_sec": 1.0,
        "max_frames_signif": 5,
        "count_nonsignif": 2,
        "hires": False,
        "small_video": False,  # SECURITY 下禁止小视频
        "small_video_encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"},
    },
    MODEL.OFFLINE: {
        "interval_sec": 1.0,
        "max_frames_nonsignif": 6,
        "hires": False,
        "small_video": True,
        "small_video_encode": {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"},
    },
}


def decide_vlm_sampling(mode: MODEL, significant_motion: bool) -> Dict:
    cfg = VLM_POLICY[mode]
    policy = {
        "extract_keyframes": False,
        "interval_sec": cfg.get("interval_sec", 1.0),
        "max_frames": None,
        "use_fixed_sampling": False,
        "fixed_count": 0,
        "emit_small_video": False,
        "encode": cfg.get("small_video_encode", {"fps": 8, "height": 480, "crf": 28, "preset": "veryfast"}),
        "hires": cfg.get("hires", False),
    }
    if mode == MODEL.ONLINE:
        policy["extract_keyframes"] = True
        policy["max_frames"] = cfg.get("max_frames", 3)
    elif mode == MODEL.SECURITY:
        if significant_motion:
            policy["extract_keyframes"] = True
            policy["max_frames"] = cfg.get("max_frames_signif", 5)
        else:
            policy["use_fixed_sampling"] = True
            policy["fixed_count"] = cfg.get("count_nonsignif", 2)
    else:  # OFFLINE
        if significant_motion:
            policy["emit_small_video"] = cfg.get("small_video", True)
        else:
            policy["extract_keyframes"] = True
            policy["max_frames"] = cfg.get("max_frames_nonsignif", 6)
    return policy


def _safe_put_with_ctrl(
    q: Optional[Queue], obj: dict, q_ctrl: Queue, stop: object,
    *, timeout=0.2, max_tries=50
) -> bool:
    if q is None:
        return False

    tries = 0
    while True:
        # ---- 控制优先：收到 STOP 立即放弃本次投递 ----
        try:
            msg = q_ctrl.get_nowait()
            if msg is stop or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                logger.info("[A] 检测到控制队列 STOP，停止将本次切片传递到数据队列。")
                return False
            try:
                q_ctrl.put_nowait(msg)
            except Full:
                pass
        except Empty:
            pass

        # ---- 尝试 put ----
        try:
            q.put(obj, timeout=timeout)
            return True
        except Full:
            tries += 1
            if tries >= max_tries:
                logger.warning("[A] 队列持续拥堵，放弃投递。")
                return False
        except Exception as e:
            logger.warning(f"[A] put 发生异常，将视为失败：{e}")
            return False


# -------------------- A 线程主体 --------------------
def worker_a_cut(
    url: str,
    have_audio_track: bool,
    mode: MODEL,
    slice_sec: int,
    q_audio: Optional[Queue],   # <- 可选队列
    q_video: Optional[Queue],   # <- 可选队列
    q_ctrl: Queue,
    stop: object,
):
    """
    A 侧流程总览（切片 → 标准化 → 取帧/小视频 → 兜底）【运维备注】

    一、时间窗策略
    - 由 (_mode_window_policy) 基于 mode 与 slice_sec 计算 (win, step)：
    SECURITY：win∈[4,12]，step=win（默认建议 4~8s，实时告警取短窗）
    ONLINE  ：win∈[5,16]，step=win（默认建议 5~10s，面向直播看板）
    OFFLINE ：win∈[8,30]，step=win-overlap（默认建议 8~12s，可轻重叠）
    - t0 按 step 递增；离线到 EOF 做尾窗兜底后退出；实时常驻。

    二、切片与标准化（FFmpeg，无中间临时文件）
    - 函数：FFmpegUtils.cut_and_standardize_segment(src_url, t0, win, …)
    1) 探测源是否有视频/音频流（ffprobe）
    2) 视频：先尝试「流拷贝」(-c:v copy, -an) 输出无声 mp4；
        若容器/时间基异常导致失败 → 自动回退「重编码」输出；
    3) 音频：恒定重采样为 16k/mono/PCM16 的 wav
    - 返回：{"video_path"|None, "audio_path"|None, "t0","t1","duration","index","have_audio"}
    - 允许失败：若源/段损坏，可能返回 None 或空文件；上游必须检查存在且 size>0。

    三、运动检测（轻量、仅用于策略选择）
    - fast_motion_detect_placeholder：1Hz 采样 + 邻帧差分均值
    - 输出布尔 significant_motion，用于决定 VLM 取帧策略（不影响切片本身）

    四、取帧/小视频策略（按模式与运动情况）
    - 策略选择（decide_vlm_sampling）：
    SECURITY：无显著运动 → 固定抽帧；有显著运动 → 间隔关键帧
    ONLINE  ：恒走「间隔关键帧」
    OFFLINE ：显著运动 → 允许小视频；否则走「间隔关键帧」
    - 小视频仅在 OFFLINE 允许（SECURITY/ONLINE 强制禁用）：
    - compress_video_for_vlm(in→out, fps/height/crf/preset)
    - 失败则回退到图像策略

    五、抽帧实现与多级兜底（关键！）
    1) 固定抽帧（sample_fixed_frames_with_pts）
    优先级：SECURITY(无显著) > OFFLINE(无小视频且不显著) 可能使用
    路线：
    a. OpenCV「顺播」读取，按目标索引（等分全片）命中即保存
    b. 若 OpenCV 打不开或顺播失败 → FFmpeg 兜底
        - 按帧索引近似导出（select='eq(n,idx)'），命不中则跳过该张
        - 再不行 → 按时间等间隔 fps=1/interval 抓帧
    c. 返回 (paths, pts≈t0+idx/fps, frame_indices)；兜底场景下 pts/idx 可能近似

    2) 间隔关键帧（extract_keyframes_by_interval_with_pts）
    优先级：ONLINE；SECURITY(显著运动)；OFFLINE(不显著且禁小视频)
    路线：
    a. OpenCV：每 interval 取一帧，做相邻差分（阈值 0.65）筛“变化大”的帧
    b. OpenCV 打不开 → FFmpeg 兜底（fps=1/interval 等间隔导出）
    c. 同样返回 (paths, pts≈t0+idx/fps, frame_indices)；兜底时为近似值

    3) 小视频（仅 OFFLINE & 显著运动）
    a. 压缩成功 → small_video_path + fps
    b. 失败 → 回落到（1）或（2）的图像路径

    六、队列投递与跳过条件
    - 仅当「有视频」且（small_video_path 或 keyframes 非空）时才投 q_video
    - 有音轨且需要 ASR 时投 q_audio（附带静音探测提示）
    - 若视频抽帧/小视频全失败 → 记录 WARNING 并跳过该段（不投 B）

    七、时间戳与对齐
    - 每段携带 clip_t0/clip_t1（=t0/t1），用于下游统一时间线
    - 每张图像附带 frame_pts≈t0+idx/fps（兜底路径为近似）

    八、日志与排障要点
    - 关键 INFO：
    [A] seg#N mode=… signif=… policy: emit_small_video=…, use_fixed_sampling=…, extract_keyframes=…
    - 关键 WARNING：
    - “待抽取固定帧视频，无法通过 OpenCV 打开，使用 FFmpeg 兜底”
    - “固定抽帧未取到图片，回落到按间隔关键帧策略/FFmpeg 抓帧”
    - “待抽取关键帧视频，无法通过 OpenCV 打开，使用 FFmpeg 兜底”
    - “无可用的小视频/关键帧文件，跳过该段”
    - 常见外部原因：RTSP 不支持 PAUSE、H264 比特流损坏、B 帧/时间基异常导致 seek 困难

    九、参数建议
    - slice_sec：SECURITY/ONLINE 取 5~10s；OFFLINE 8~12s（可小重叠）
    - diff_threshold：0.65（图像差分）；motion diff_threshold：15.0
    - JPG 质量 90；关键帧 interval 默认 1.0s（按需调整）

    （本说明仅描述 A 侧，B/C 侧的节流/对齐由主控与 skew_guard 决定）
    """

    running = False
    paused = False
    cur_mode = mode
    desire_win = max(1, int(slice_sec))
    overlap_sec = 0.0  # offline only
    t0 = 0.0
    seg_idx = 0
    out_dir = "out"

    # 离线有总时长；RTSP/直播 None
    max_duration: Optional[float] = None
    try:
        max_duration = FFmpegUtils._probe_duration_seconds(url)
    except Exception:
        logger.warning("[A] FFmpeg 获取媒体总时长失败，按实时流常驻处理")
    tail_emitted = False  # offline 尾窗兜底只发一次

    def drain_ctrl_once() -> Optional[str]:
        nonlocal running, paused, cur_mode, desire_win, overlap_sec
        try:
            msg = q_ctrl.get_nowait()
        except Empty:
            return None

        if msg is stop:
            return "STOP"

        if isinstance(msg, dict):
            typ = msg.get("type")
            if typ == "START":
                logger.info("[A] 收到 START，开始切片")
                running = True
            elif typ == "PAUSE":
                logger.info("[A] 收到 PAUSE，暂停切片")
                paused = True
            elif typ == "RESUME":
                logger.info("[A] 收到 RESUME，继续切片")
                paused = False
            elif typ == "MODE_CHANGE":
                val = msg.get("value")
                try:
                    # 既兼容传入枚举，也兼容传入字符串 value
                    cur_mode = val if isinstance(val, MODEL) else MODEL(str(val))
                    logger.info(f"[A] 模式切换为：{cur_mode}")
                except Exception:
                    logger.warning("[A] MODE_CHANGE 值无法解析：%r", val)
            elif typ == "UPDATE_SLICE":
                try:
                    desire_win = max(1, int(msg.get("value", desire_win)))
                    logger.info(f"[A] 切窗大小更新：{desire_win}")
                except Exception:
                    pass
            elif typ == "UPDATE_OVERLAP":
                try:
                    overlap_sec = float(msg.get("value", overlap_sec))
                    logger.info(f"[A] 窗口重叠更新：{overlap_sec}")
                except Exception:
                    pass
            elif typ in ("STOP", "SHUTDOWN"):
                return "STOP"
        return None

    try:
        while True:
            # 控制优先，降低延迟
            for _ in range(4):
                res = drain_ctrl_once()
                if res == "STOP":
                    logger.info("[A] 收到控制队列 STOP，退出")
                    return
                if not running or paused:
                    sleep(0.01)

            if not running or paused:
                continue

            # 窗口/步长（含 offline 重叠）
            win, step = _mode_window_policy(cur_mode, desire_win, overlap_sec)

            # 离线 EOF
            if max_duration is not None and t0 >= max_duration:
                if cur_mode == MODEL.OFFLINE and not tail_emitted and max_duration > 0:
                    new_t0 = max(0.0, max_duration - win)
                    if new_t0 + 1e-6 < t0:
                        t0 = new_t0
                        tail_emitted = True
                    else:
                        logger.info("[A] 离线已切完，退出")
                        return
                else:
                    logger.info("[A] 离线已切完，退出")
                    return

            # ---- 切窗+标准化 ----
            seg = FFmpegUtils.cut_and_standardize_segment(
                src_url=url, start_time=t0, duration=win,
                output_dir=out_dir, segment_index=seg_idx, have_audio=have_audio_track
            )
            v_out = seg.get("video_path")  # 可能为 None（纯音频源或导出失败）
            a_out = seg.get("audio_path")  # 可能为 None（纯视频源或无音轨）
            has_v = _file_exists_nonzero(v_out)
            has_a = _file_exists_nonzero(a_out)

            # 是否产出音频
            produce_audio = bool(have_audio_track and q_audio is not None and has_a)

            # 静音提示
            if produce_audio:
                silence_hint = silence_detect_hint(a_out)
            else:
                silence_hint = {"silence_ratio": 0.0, "is_mostly_silent": False, "segments": []}

            # 运动 & 关键帧策略（仅当有视频且启用 B）
            keyframes: List[str] = []
            frame_pts: List[float] = []
            frame_indices: List[int] = []
            small_video_path: Optional[str] = None
            small_video_fps: Optional[float] = None
            policy_used = "none"
            hires_flag = False

            if has_v and q_video is not None:
                signif = fast_motion_detect_placeholder(
                    v_out, sample_interval=1.0, diff_threshold=15.0, max_samples=30
                )
                vlm_policy = decide_vlm_sampling(cur_mode, signif)
                hires_flag = vlm_policy["hires"]
                tag = f"seg{seg_idx:04d}"

                # SECURITY/ONLINE 不会走小视频（配置已禁用）
                if vlm_policy["emit_small_video"]:
                    small_video_path = os.path.join(out_dir, f"{tag}_small.mp4")
                    enc = vlm_policy["encode"]
                    try:
                        FFmpegUtils.compress_video_for_vlm(
                            in_video=v_out, out_video=small_video_path,
                            fps=enc.get("fps", 8), height=enc.get("height", 480),
                            crf=enc.get("crf", 28), preset=enc.get("preset", "veryfast")
                        )
                        small_video_fps, _ = _get_video_meta(small_video_path)
                        policy_used = "small_video"
                    except Exception as e:
                        logger.warning("[A] 小视频压缩失败，将回落到图像策略：%s", e)
                        small_video_path = None
                        small_video_fps = None

                if not small_video_path:
                    # 固定抽帧优先（更稳定），失败再回落到“间隔关键帧”
                    imgs, pts, idxs = [], [], []
                    if vlm_policy.get("use_fixed_sampling"):
                        imgs, pts, idxs = sample_fixed_frames_with_pts(
                            v_out, out_dir, tag,
                            count=int(vlm_policy.get("fixed_count") or 2),
                            jpg_quality=90, seg_t0=seg["t0"]
                        )
                        if imgs:
                            policy_used = "fixed_sampling"
                        else:
                            logger.warning("[A] seg#%s 固定抽帧未取到图片，回落到按间隔关键帧策略。", seg_idx)

                    if not imgs and vlm_policy.get("extract_keyframes"):
                        dynamic_max = vlm_policy.get("max_frames")
                        imgs, pts, idxs = extract_keyframes_by_interval_with_pts(
                            v_out, out_dir, tag,
                            interval_sec=float(vlm_policy["interval_sec"]),
                            diff_threshold=0.65,
                            max_frames=dynamic_max,
                            seg_t0=seg["t0"]
                        )
                        if imgs:
                            policy_used = "keyframes"

                    keyframes, frame_pts, frame_indices = imgs, pts, idxs

                # 兜底：若小视频被禁用且又没抽到任何图片→跳过视频侧
                if not small_video_path and not keyframes:
                    logger.warning("[A] seg#%s 无可用的小视频/关键帧文件，跳过该段。", seg_idx)

                logger.info(
                    "[A] seg#%s mode=%s signif=%s policy: emit_small_video=%s, use_fixed_sampling=%s, extract_keyframes=%s",
                    seg_idx, cur_mode.value, bool(signif),
                    bool(vlm_policy.get("emit_small_video")),
                    bool(vlm_policy.get("use_fixed_sampling")),
                    bool(vlm_policy.get("extract_keyframes")),
                )
            else:
                signif = False

            # ---- 投视频队列 ----
            if has_v and q_video is not None and (small_video_path or keyframes):
                payload_video = {
                    "path": v_out,
                    "t0": seg["t0"],
                    "t1": seg["t1"],
                    "clip_t0": seg["t0"],
                    "clip_t1": seg["t1"],
                    "segment_index": seg_idx,
                    "mode": cur_mode.value,
                    "win": win,
                    "step": step,
                    "overlap_sec": overlap_sec if cur_mode == MODEL.OFFLINE else 0.0,
                    "has_audio": has_a,
                    "keyframes": keyframes,
                    "frame_pts": frame_pts,
                    "frame_indices": frame_indices,
                    "keyframe_count": len(keyframes),
                    "small_video": small_video_path,
                    "small_video_fps": small_video_fps,
                    "hires": hires_flag,
                    "policy": {
                        "significant_motion": bool(signif),
                        "policy_used": (policy_used or "none"),
                        "interval_sec": (float(vlm_policy["interval_sec"]) if has_v and q_video is not None else 0.0),
                        "max_frames": (vlm_policy.get("max_frames") if has_v and q_video is not None else None),
                        "silence_hint": {
                            "silence_ratio": float(silence_hint.get("silence_ratio", 0.0)),
                            "is_mostly_silent": bool(silence_hint.get("is_mostly_silent", False)),
                        },
                    },
                }
                ok = _safe_put_with_ctrl(q_video, payload_video, q_ctrl, stop)
                if not ok:
                    return  # 退出 A 线程

            # ---- 投音频队列 ----
            if produce_audio and q_audio is not None:
                audio_payload = {
                    "path": a_out,
                    "t0": seg["t0"],
                    "t1": seg["t1"],
                    "segment_index": seg_idx,
                    "silence_hint": {
                        "silence_ratio": float(silence_hint.get("silence_ratio", 0.0)),
                        "is_mostly_silent": bool(silence_hint.get("is_mostly_silent", False)),
                    },
                }
                ok = _safe_put_with_ctrl(q_audio, audio_payload, q_ctrl, stop)
                if not ok:
                    return

            # 步进
            seg_idx += 1
            t0 += step
            if tail_emitted and max_duration is not None:
                t0 = max_duration

            sleep(0.005)

    except Exception as e:
        logger.error(f"[A] 运行时异常，线程退出：{e}")
    finally:
        logger.info('[A] 线程退出清理完成')
