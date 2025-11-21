# -*- coding: utf-8 -*-
"""
约定：
1) 所有 ffmpeg/ffprobe 子进程都通过 ffbin()/fpbin() 获取绝对路径。
2) RTSP 输入统一附加：-nostdin -rtsp_transport tcp -hwaccel none -c:v h264
   以及低延迟/小探测缓冲：-fflags nobuffer -flags low_delay -probesize 256k -analyzeduration 0
   并开启“到达壁钟映射”：-use_wallclock_as_timestamps 1
3) 切片流程：优先 copy，OpenCV 校验失败则重编码（libx264 + yuv420p）。
4) ensure_ffmpeg() 在启动时打印并校验实际使用的二进制绝对路径；发现可疑（bm/sophon）版本直接警告或中止。

说明（壁钟映射）：
- 我们使用输入参数 -use_wallclock_as_timestamps 1，让解复用时刻以“本机壁钟到达时间”为参考；
- 在 cut_and_standardize_segment() 中记录窗口开始（调用时刻）的 epoch，并生成 t0_iso/t1_iso；
- 这是“到达时间”，会包含网络/缓冲延迟，与“摄像机 UTC 拍摄时刻”略有差异，但实现成本最低、展示体验好。
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from typing import Optional, Dict, List, Iterator
from time import time
from datetime import datetime, timezone

import cv2

from src.utils.logger_utils import get_logger

logger = get_logger(__name__)

__all__ = [
    "ffbin", "fpbin", "ensure_ffmpeg",
    "normalize_path", "is_rtsp", "input_args_for",
    "run_ffprobe", "probe_duration_seconds",
    "get_audio_duration_seconds", "get_video_duration_seconds",
    "have_audio_track", "standardize_audio_to_wav16k_mono",
    "standardize_video_strip_audio", "compress_video_for_vlm",
    "cut_and_standardize_segment", "streaming_cut_generator",
    "detect_silence_intervals",
    "grab_frame_by_index",
]

# ============================================================
# 在 Python 进程内部初始化 ffmpeg 环境变量
# （等价于在 shell 里运行：
#   export PATH=".../envs/ffmpeg-opencv/bin:$PATH"
#   export FFMPEG_BIN=".../envs/ffmpeg-opencv/bin/ffmpeg"
#   export FFPROBE_BIN=".../envs/ffmpeg-opencv/bin/ffprobe"
# 但作用范围仅限当前 Python 进程及其子进程）
# ============================================================

_FFMPEG_ENV_BIN_DIR = "/home/linaro/miniconda3/envs/ffmpeg-opencv/bin"
_FFMPEG_DEFAULT_BIN = os.path.join(_FFMPEG_ENV_BIN_DIR, "ffmpeg")
_FFPROBE_DEFAULT_BIN = os.path.join(_FFMPEG_ENV_BIN_DIR, "ffprobe")


def _bootstrap_ffmpeg_env() -> None:
    """
    在 Python 进程内部初始化 ffmpeg 环境变量：

    - 如果 _FFMPEG_ENV_BIN_DIR 存在：
        * 把该目录插入 PATH 最前面；
        * 若外部没有显式设置 FFMPEG_BIN / FFPROBE_BIN，则设置默认值；
    - 如果目录不存在（如开发机上没有这个 env），则不做任何修改，
      此时 ffbin()/fpbin() 会退回使用系统默认 ffmpeg/ffprobe。
    """
    if not os.path.isdir(_FFMPEG_ENV_BIN_DIR):
        # 非盒子环境或该 env 不存在：保持系统默认行为
        return

    old_path = os.environ.get("PATH", "")
    # 防止重复插入
    if _FFMPEG_ENV_BIN_DIR not in old_path.split(":"):
        os.environ["PATH"] = f"{_FFMPEG_ENV_BIN_DIR}:{old_path}"

    os.environ.setdefault("FFMPEG_BIN", _FFMPEG_DEFAULT_BIN)
    os.environ.setdefault("FFPROBE_BIN", _FFPROBE_DEFAULT_BIN)


# 模块导入时自动执行一次环境注入
_bootstrap_ffmpeg_env()

# ============================================================

SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")


def grab_frame_by_index(video_path: str, out_dir: str, out_name: str, index: int) -> List[str]:
    """
    近似按帧索引导出 1 张图片（兜底/应急）。
    - 使用系统/环境指定的 ffmpeg，可与 sophon-ffmpeg_0.10.0 兼容。
    - out_name 不含扩展名，函数会生成 .jpg。

    返回：成功时 [输出路径]；失败时 []。
    """
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{out_name}.jpg")
    cmd = [
        ffbin(), "-y", "-hide_banner", "-loglevel", "error",
        "-i", video_path, "-vf", f"select='eq(n\\,{max(0, index)})'", "-vsync", "vfr", out
    ]
    try:
        subprocess.check_call(cmd)
        return [out] if (os.path.exists(out) and os.path.getsize(out) > 0) else []
    except Exception:
        return []


# ======== 小工具 ========

def _iso_utc(ts_epoch: float) -> str:
    """epoch 秒 -> ISO-8601（UTC，秒级，尾部 Z）"""
    try:
        return datetime.fromtimestamp(float(ts_epoch), tz=timezone.utc) \
            .isoformat(timespec="seconds") \
            .replace("+00:00", "Z")
    except Exception:
        return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ======== 路径与输入参数 ========

def ffbin() -> str:
    """优先用环境变量指定的 ffmpeg；否则查 PATH。"""
    env_bin = os.getenv("FFMPEG_BIN")
    if env_bin:
        return env_bin
    found = shutil.which("ffmpeg")
    return found or "ffmpeg"


def fpbin() -> str:
    """优先用环境变量指定的 ffprobe；否则查 PATH。"""
    env_bin = os.getenv("FFPROBE_BIN")
    if env_bin:
        return env_bin
    found = shutil.which("ffprobe")
    return found or "ffprobe"


def ensure_ffmpeg() -> None:
    ff = ffbin()
    fp = fpbin()
    if not shutil.which(ff):
        raise RuntimeError(f"[FFmpeg] 未找到 ffmpeg，可导出 FFMPEG_BIN 指定：{ff}")
    if not shutil.which(fp):
        raise RuntimeError(f"[FFmpeg] 未找到 ffprobe，可导出 FFPROBE_BIN 指定：{fp}")
    logger.info(f"[FFmpeg] using ffmpeg={shutil.which(ff)} ffprobe={shutil.which(fp)}")


def normalize_path(url_or_path: str) -> str:
    """将 file://xxx 转为本地路径；其他协议原样返回"""
    if (url_or_path or "").startswith("file://"):
        return url_or_path.replace("file://", "", 1)
    return url_or_path


def is_rtsp(src: str) -> bool:
    s = (src or "").lower()
    return s.startswith("rtsp://") or s.startswith("rtsps://")


def input_args_for(src: str, *, for_probe: bool = False) -> List[str]:
    """
    为 RTSP / 本地分别返回合适的输入端参数。
    for_probe=True 时用于 ffprobe（ffprobe 不支持 -hwaccel / -c:v / -nostdin）。
    这里为 RTSP 输入统一加 -use_wallclock_as_timestamps 1。
    """
    rtsp = is_rtsp(src)
    if for_probe:
        return ["-rtsp_transport", "tcp", "-use_wallclock_as_timestamps", "1"] if rtsp else []
    if rtsp:
        return [
            "-nostdin",
            "-rtsp_transport", "tcp",
            "-use_wallclock_as_timestamps", "1",   # ★ 壁钟映射
            "-hwaccel", "none",
            "-c:v", "h264",
            "-fflags", "nobuffer",
            "-flags", "low_delay",
            "-probesize", "256k",
            "-analyzeduration", "0",
        ]
    return ["-nostdin"]


# ======== ffprobe 基础 ========

def run_ffprobe(cmd: List[str]) -> str:
    return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)


def probe_duration_seconds(url_or_path: str) -> Optional[float]:
    """
    通用时长探测：本地文件返回秒数；RTSP/直播可能无 duration -> 返回 None
    """
    src = normalize_path(url_or_path)
    in_args = input_args_for(src, for_probe=True)

    # 方式A：简单输出
    try:
        out = run_ffprobe([
            fpbin(), "-v", "error", *in_args,
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            src
        ]).strip()
        if out:
            return max(0.0, float(out))
    except Exception:
        pass

    # 方式B：JSON 回退
    try:
        jout = run_ffprobe([
            fpbin(), "-v", "error", *in_args,
            "-print_format", "json",
            "-show_format",
            src
        ])
        j = json.loads(jout)
        dur = j.get("format", {}).get("duration")
        if dur is None or dur == "N/A":
            return None
        return max(0.0, float(dur))
    except Exception as e:
        logger.debug(f"[FFmpeg] ffprobe 获取时长失败: {e}")
        return None


# ======== 对外简单 API ========

def get_audio_duration_seconds(standardized_audio: str) -> Optional[float]:
    return probe_duration_seconds(standardized_audio)


def get_video_duration_seconds(standardized_video: str) -> Optional[float]:
    return probe_duration_seconds(standardized_video)


# ======== 音轨判定 ========

def have_audio_track(url_or_path: str) -> bool:
    src = normalize_path(url_or_path)
    in_args = input_args_for(src, for_probe=True)
    try:
        jout = run_ffprobe([
            fpbin(), "-v", "error", *in_args,
            "-select_streams", "a", "-show_streams",
            "-print_format", "json", src
        ])
        j = json.loads(jout)
        streams = j.get("streams", [])
        logger.info(f'[ffmpeg] 音轨探测 have_audio_track(): {len(streams) > 0}')
        return len(streams) > 0
    except Exception as e:
        logger.debug(f"[FFmpeg] ffprobe 检测音轨失败: {e}")
        return False


# ======== 标准化 ========

def standardize_audio_to_wav16k_mono(src_path: str, out_path: str) -> str:
    ensure_ffmpeg()
    src = normalize_path(src_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cmd = [
        ffbin(), "-y",
        *input_args_for(src),
        "-i", src, "-vn",
        "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
        out_path
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path


def standardize_video_strip_audio(src_path: str, out_path: str, reencode: bool = False) -> str:
    """
    生成无声视频给视觉侧：
    - reencode=False（默认）：视频流直接拷贝，最快
    - reencode=True：重编码为 H.264 + yuv420p，兼容性更好但耗时更长
    """
    ensure_ffmpeg()
    src = normalize_path(src_path)
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    if not reencode:
        cmd = [
            ffbin(), "-y",
            *input_args_for(src),
            "-i", src, "-an", "-c:v", "copy",
            out_path
        ]
    else:
        cmd = [
            ffbin(), "-y",
            *input_args_for(src),
            "-i", src, "-an",
            "-c:v", "libx264", "-preset", "fast", "-crf", "23",
            "-pix_fmt", "yuv420p",
            out_path
        ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_path


# ======== 小视频压缩（给 VLM / 直传场景） ========

def compress_video_for_vlm(
    in_video: str, out_video: str,
    fps: int = 8, height: int = 480,
    crf: int = 28, preset: str = "veryfast"
) -> str:
    """
    小视频压缩规范：
    - 编码：H.264
    - 分辨率：最长边 ≤ height（等比缩放：scale=-2:height）
    - FPS <= fps
    - CRF ~ 28，preset veryfast
    - 去音轨（ASR 走单独管线）
    """
    ensure_ffmpeg()
    src = normalize_path(in_video)
    os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
    vf = f"scale=-2:{int(height)}"
    cmd = [
        ffbin(), "-y",
        *input_args_for(src),
        "-i", src, "-an",
        "-r", str(int(fps)), "-vf", vf,
        "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        out_video
    ]
    subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    return out_video


# ======== 切窗并标准化（含容错） ========

def cut_and_standardize_segment(
    src_url: str, start_time: float, duration: float,
    output_dir: str, segment_index: int, have_audio: bool = True
) -> Dict[str, Optional[str]]:
    """
    切出一段音/视频片段，并标准化为：
    - 若有视频：去音轨的无声 mp4（video_path）
    - 若有音频：16kHz 单声道 wav（audio_path）
    纯音频 → 仅返回 audio_path（video_path=None）
    纯视频 → 仅返回 video_path（audio_path=None）

    容错：
    1) 先用“流拷贝”快速切，写完用 OpenCV 校验能否读取；
    2) 若打不开/0帧，自动回退“重编码”切片，确保可读。

    附加（壁钟映射）：
    - 在调用开始时记录 wallclock_epoch = time()，并生成 t0_iso / t1_iso（用于前端“RTSP 中的近似真实时间”展示）
    """
    import cv2 as _cv2  # 延迟导入，避免环境缺少时影响其余功能

    def _verify_openable(p: str) -> bool:
        try:
            cap = _cv2.VideoCapture(p)
            if not cap.isOpened():
                cap.release()
                return False
            total = int(cap.get(_cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total <= 0:
                ok, _ = cap.read()
                cap.release()
                return bool(ok)
            cap.release()
            return True
        except Exception:
            return False

    ensure_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    # 记录窗口壁钟锚点（近似到达时间）
    wallclock_epoch_t0 = time()
    wallclock_epoch_t1 = wallclock_epoch_t0 + float(duration)

    # 目标路径
    v_out = os.path.join(output_dir, f"segment_{segment_index:04d}_video.mp4")
    a_out = os.path.join(output_dir, f"segment_{segment_index:04d}_audio.wav")

    src_norm = normalize_path(src_url)

    # 探测实际存在的音/视频流（不要只信 have_audio）
    def _has_stream(kind: str) -> bool:
        # kind: 'v' or 'a'
        cmd = [
            fpbin(), "-v", "error",
            *input_args_for(src_norm, for_probe=True),
            "-select_streams", f"{kind}:0",
            "-show_entries", "stream=index",
            "-of", "json",
            src_norm,
        ]
        try:
            p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
            data = json.loads(p.stdout or "{}")
            return bool(data.get("streams"))
        except Exception:
            return False

    has_video = _has_stream("v")
    has_audio_real = _has_stream("a")

    v_path_ret: Optional[str] = None
    a_path_ret: Optional[str] = None

    # -------- 视频 --------
    if has_video:
        # 尝试 1：流拷贝（快）
        cmd_v_fast = [
            ffbin(), "-y", "-hide_banner", "-loglevel", "error",
            *input_args_for(src_norm),
            "-ss", str(start_time), "-t", str(duration), "-i", src_norm,
            "-map", "v:0", "-an",
            "-fflags", "+genpts",
            "-avoid_negative_ts", "make_zero",
            "-movflags", "+faststart",
            "-c:v", "copy",
            v_out
        ]
        try:
            subprocess.check_call(cmd_v_fast)
        except subprocess.CalledProcessError:
            logger.warning('[FFmpeg] 流拷贝切分失败, 回落到重编码切分。')
            # 直接重编码
            cmd_v_slow = [
                ffbin(), "-y", "-hide_banner", "-loglevel", "error",
                *input_args_for(src_norm),
                "-i", src_norm, "-ss", str(start_time), "-t", str(duration),
                "-map", "v:0", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-g", "16", "-keyint_min", "16", "-sc_threshold", "0",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                v_out
            ]
            subprocess.check_call(cmd_v_slow)
        else:
            # 写成功后验证可读性
            if not _verify_openable(v_out):
                logger.warning('[FFmpeg] 流拷贝切片视频无法被OpenCV打开, 删除坏文件, 回落到重编码切分。')
                try:
                    os.remove(v_out)
                except Exception:
                    pass
                cmd_v_slow = [
                    ffbin(), "-y", "-hide_banner", "-loglevel", "error",
                    *input_args_for(src_norm),
                    "-i", src_norm, "-ss", str(start_time), "-t", str(duration),
                    "-map", "v:0", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-g", "16", "-keyint_min", "16", "-sc_threshold", "0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    v_out
                ]
                subprocess.check_call(cmd_v_slow)

        v_path_ret = v_out
    else:
        v_path_ret = None

    # -------- 音频 --------
    if has_audio_real:
        cmd_a = [
            ffbin(), "-y", "-hide_banner", "-loglevel", "error",
            *input_args_for(src_norm),
            "-ss", str(start_time), "-t", str(duration), "-i", src_norm,
            "-map", "a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
            a_out
        ]
        subprocess.check_call(cmd_a)
        a_path_ret = a_out
    else:
        a_path_ret = None

    logger.info(f'切出视频片段：{v_path_ret}, 切出音频片段：{a_path_ret}')
    return {
        "video_path": v_path_ret,
        "audio_path": a_path_ret,
        "t0": start_time,
        "t1": start_time + duration,
        "duration": duration,
        "index": segment_index,
        "have_audio": bool(a_path_ret),  # 用“实际产出”作为 has_audio
        # 壁钟映射（到达时间 → 近似“真实时间”）
        "t0_iso": _iso_utc(wallclock_epoch_t0),
        "t1_iso": _iso_utc(wallclock_epoch_t1),
        "t0_epoch": wallclock_epoch_t0,
        "t1_epoch": wallclock_epoch_t1,
    }


# ======== 连续切窗生成器 ========

def streaming_cut_generator(
    src_url: str, output_dir: str,
    slice_sec: int = 10, have_audio: Optional[bool] = None,
    start_offset: float = 0.0, max_duration: Optional[float] = None,
) -> Iterator[Dict[str, Optional[str]]]:
    """
    连续流式切窗（生成标准化后片段）：
    - 本地文件：探测总时长，到末尾停止
    - RTSP/直播：无限循环（外部 STOP 中断）
    - 片段内的音/视频轨道是否存在，以 cut_and_standardize_segment 的“实际产出”为准
    """
    ensure_ffmpeg()
    os.makedirs(output_dir, exist_ok=True)

    # 离线文件尝试探测总时长
    total_dur = probe_duration_seconds(src_url)
    if total_dur is not None and max_duration is None:
        max_duration = total_dur

    seg_idx = 0
    t0 = float(start_offset)

    logger.info(f"开始流式切窗: {src_url}, 窗口={slice_sec}s")
    while True:
        if max_duration is not None and t0 >= max_duration:
            logger.info(f"[FFmpeg] 已到达文件末尾，总时长 {max_duration:.2f}s，结束切窗。")
            break

        seg = cut_and_standardize_segment(
            src_url=src_url, start_time=t0,
            duration=slice_sec, output_dir=output_dir,
            segment_index=seg_idx, have_audio=True  # 实际以探测为准
        )

        has_v = bool(seg.get("video_path"))
        has_a = bool(seg.get("audio_path"))
        logger.debug(
            "[FFmpeg] 切片完成 seg#%04d t0=%.3f t1=%.3f 产出: video=%s audio=%s | t0_iso=%s",
            seg_idx, seg["t0"], seg["t1"],
            "yes" if has_v else "no",
            "yes" if has_a else "no",
            seg.get("t0_iso"),
        )

        yield seg

        seg_idx += 1
        t0 += slice_sec


# ======== 静音探测 ========

def detect_silence_intervals(
    src_url: str,
    noise_db: float = -35.0,
    min_silence: float = 0.5,
    audio_stream_index: Optional[int] = None,
    timeout_sec: Optional[float] = None,
    max_intervals: Optional[int] = None,
) -> List[Dict[str, Optional[float]]]:
    """
    基于 FFmpeg silencedetect 的静音探测。
    适用于本地文件与 RTSP/直播流（直播建议设置 timeout_sec 以避免无限跑）。

    返回：
    [
        {"start": 12.34, "end": 14.90, "duration": 2.56},
        ...
    ]
    若流在“静音中”就结束，最后一段的 end/duration 可能为 None。
    """

    def _normalize(p: str) -> str:
        return p.replace("file://", "", 1) if p.startswith("file://") else p

    src = _normalize(src_url)

    filter_expr = f"silencedetect=noise={noise_db}dB:d={min_silence}"
    cmd = [
        ffbin(), "-hide_banner", "-nostats", "-v", "info",
        *input_args_for(src),
        "-i", src,
    ]
    if audio_stream_index is not None:
        cmd += ["-map", f"0:a:{audio_stream_index}"]
    else:
        cmd += ["-map", "0:a:0"]
    cmd += ["-af", filter_expr, "-vn", "-f", "null", "-"]

    proc = subprocess.Popen(
        cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1
    )

    intervals: List[Dict[str, Optional[float]]] = []
    cur_start: Optional[float] = None
    start_ts = time()

    try:
        assert proc.stderr is not None
        for line in iter(proc.stderr.readline, ''):
            if timeout_sec is not None and (time() - start_ts) > timeout_sec:
                break

            m1 = SILENCE_START_RE.search(line)
            if m1:
                try:
                    cur_start = float(m1.group(1))
                except Exception:
                    cur_start = None
                continue

            m2 = SILENCE_END_RE.search(line)
            if m2:
                try:
                    end = float(m2.group(1))
                    dur = float(m2.group(2))
                except Exception:
                    end, dur = None, None

                if cur_start is None and end is not None and dur is not None:
                    cur_start = max(0.0, end - dur)

                intervals.append({"start": cur_start, "end": end, "duration": dur})
                cur_start = None

                if max_intervals is not None and len(intervals) >= max_intervals:
                    break

        try:
            proc.terminate()
        except Exception:
            pass
        proc.wait(timeout=2)
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
        raise
    finally:
        if cur_start is not None:
            intervals.append({"start": cur_start, "end": None, "duration": None})

    return intervals
