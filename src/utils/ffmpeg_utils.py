'''
Author: 13594053100@163.com
Date: 2025-09-30 09:46:18
LastEditTime: 2025-10-14 14:54:35
'''

# -*- coding: utf-8 -*-
"""
FFmpegUtils：ffmpeg 相关工具
- ensure_ffmpeg / have_audio_track / _probe_duration_seconds
- 标准化：standardize_audio_to_wav16k_mono / standardize_video_strip_audio
- 切窗并标准化：cut_and_standardize_segment
- 连续切窗生成器：streaming_cut_generator
- 新增：compress_video_for_vlm（小视频压缩）
"""

import json
import os
import re
import shutil
import subprocess
from typing import Optional, Dict, List
from pathlib import Path
from time import time

from src.utils.logger_utils import get_logger
logger = get_logger(__name__)

SILENCE_START_RE = re.compile(r"silence_start:\s*([0-9.]+)")
SILENCE_END_RE = re.compile(r"silence_end:\s*([0-9.]+)\s*\|\s*silence_duration:\s*([0-9.]+)")

class FFmpegUtils:
    """
    ffmpeg 相关工具类
    使用本类提供的方法前，必需先调用 ensure_ffmpeg()，检查是否有可用 ffmpeg 或 ffprobe
    """

    # ========= 基础工具 =========
    @staticmethod
    def ensure_ffmpeg():
        if not shutil.which("ffmpeg"):
            raise RuntimeError("[FFmpeg] ffmpeg 未在 PATH 中找到")
        if not shutil.which("ffprobe"):
            raise RuntimeError("[FFmpeg] ffprobe 未在 PATH 中找到")

    @staticmethod
    def _normalize_path(url_or_path: str) -> str:
        """将 file://xxx 转为本地路径；其他协议原样返回"""
        if url_or_path.startswith("file://"):
            return url_or_path.replace("file://", "", 1)
        return url_or_path

    @staticmethod
    def _run_ffprobe(cmd: list[str]) -> str:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True)

    @staticmethod
    def _probe_duration_seconds(url_or_path: str) -> Optional[float]:
        """
        通用时长探测：本地文件返回秒数；RTSP/直播可能无 duration -> 返回 None
        """
        src = FFmpegUtils._normalize_path(url_or_path)

        # 先用简单输出尝试
        try:
            out = FFmpegUtils._run_ffprobe([
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                src
            ]).strip()
            if out:
                return max(0.0, float(out))
        except Exception:
            pass

        # 再用 JSON 回退
        try:
            jout = FFmpegUtils._run_ffprobe([
                "ffprobe", "-v", "error",
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

    # ========= 时长获取（对外 API） =========
    @staticmethod
    def get_audio_duration_seconds(standardized_audio: str) -> Optional[float]:
        return FFmpegUtils._probe_duration_seconds(standardized_audio)

    @staticmethod
    def get_video_duration_seconds(standardized_video: str) -> Optional[float]:
        return FFmpegUtils._probe_duration_seconds(standardized_video)

    # ========= 音轨判定 =========
    @staticmethod
    def have_audio_track(url_or_path: str) -> bool:
        src = FFmpegUtils._normalize_path(url_or_path)
        try:
            jout = FFmpegUtils._run_ffprobe([
                "ffprobe", "-v", "error",
                "-select_streams", "a", "-show_streams",
                "-print_format", "json", src
            ])
            j = json.loads(jout)
            streams = j.get("streams", [])
            return len(streams) > 0
        except Exception as e:
            logger.debug(f"[FFmpeg] ffprobe 检测音轨失败: {e}")
            return False

    # ========= 标准化 =========
    @staticmethod
    def standardize_audio_to_wav16k_mono(src_path: str, out_path: str) -> str:
        FFmpegUtils.ensure_ffmpeg()
        src = FFmpegUtils._normalize_path(src_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        cmd = [
            "ffmpeg", "-y", "-i", src, "-vn",
            "-ac", "1", "-ar", "16000", "-acodec", "pcm_s16le",
            out_path
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return out_path

    @staticmethod
    def standardize_video_strip_audio(src_path: str, out_path: str, reencode: bool = False) -> str:
        """
        生成无声视频给视觉侧：
        - reencode=False（默认）：视频流直接拷贝，最快
        - reencode=True：重编码为 H.264 + yuv420p，兼容性更好但耗时更长
        """
        FFmpegUtils.ensure_ffmpeg()
        src = FFmpegUtils._normalize_path(src_path)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

        if not reencode:
            cmd = ["ffmpeg", "-y", "-i", src, "-an", "-c:v", "copy", out_path]
        else:
            cmd = [
                "ffmpeg", "-y", "-i", src, "-an",
                "-c:v", "libx264", "-preset", "fast", "-crf", "23",
                "-pix_fmt", "yuv420p", out_path
            ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return out_path

    # ========= 小视频压缩（给 VLM / 直传场景） =========
    @staticmethod
    def compress_video_for_vlm(in_video: str, out_video: str,
                               fps: int = 8, height: int = 480,
                               crf: int = 28, preset: str = "veryfast") -> str:
        """
        小视频压缩规范：
        - 编码：H.264
        - 分辨率：最长边 ≤ height（等比缩放：scale=-2:height）
        - FPS <= fps
        - CRF ~ 28，preset veryfast
        - 去音轨（ASR 走单独管线）
        """
        FFmpegUtils.ensure_ffmpeg()
        src = FFmpegUtils._normalize_path(in_video)
        os.makedirs(os.path.dirname(out_video) or ".", exist_ok=True)
        vf = f"scale=-2:{int(height)}"
        cmd = [
            "ffmpeg", "-y", "-i", src, "-an",
            "-r", str(int(fps)), "-vf", vf,
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            out_video
        ]
        subprocess.check_call(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        return out_video

    # ========= 切窗并标准化（按源轨道分别处理，无 temp.mp4；包含“打不开自动重切重编码”的容错）=========
    @staticmethod
    def cut_and_standardize_segment(
        src_url: str, start_time: float, duration: float,
        output_dir: str, segment_index: int, have_audio: bool = True
    ) -> Dict[str, str]:
        """
        切出一段音/视频片段，并标准化为：
        - 若有视频：去音轨的无声 mp4（video_path）
        - 若有音频：16kHz 单声道 wav（audio_path）
        纯音频 → 仅返回 audio_path（video_path=None）
        纯视频 → 仅返回 video_path（audio_path=None）

        容错：
        1) 先用“流拷贝”快速切，写完立即用 OpenCV 校验能否读取；
        2) 若打不开/0帧，自动回退“重编码”切片（H.264 固定 GOP/关键帧），确保可读。
        """
        import json, subprocess, os, cv2

        def _verify_openable(p: str) -> bool:
            try:
                cap = cv2.VideoCapture(p)
                if not cap.isOpened():
                    cap.release()
                    return False
                # 方式A：总帧数
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
                if total <= 0:
                    # 方式B：尝试读一帧
                    ok, _ = cap.read()
                    cap.release()
                    return bool(ok)
                cap.release()
                return True
            except Exception:
                return False

        FFmpegUtils.ensure_ffmpeg()
        os.makedirs(output_dir, exist_ok=True)

        # 目标路径
        v_out = os.path.join(output_dir, f"segment_{segment_index:04d}_video.mp4")
        a_out = os.path.join(output_dir, f"segment_{segment_index:04d}_audio.wav")

        # 探测实际存在的音/视频流（不要只信 have_audio）
        def _has_stream(kind: str) -> bool:
            # kind: 'v' or 'a'
            cmd = [
                "ffprobe", "-v", "error",
                "-select_streams", f"{kind}:0",
                "-show_entries", "stream=index",
                "-of", "json",
                FFmpegUtils._normalize_path(src_url),
            ]
            try:
                p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True)
                data = json.loads(p.stdout or "{}")
                return bool(data.get("streams"))
            except Exception:
                return False

        has_video = _has_stream("v")
        has_audio_real = _has_stream("a")

        # 通用入参
        src_norm = FFmpegUtils._normalize_path(src_url)

        v_path_ret: Optional[str] = None
        a_path_ret: Optional[str] = None

        # ------------- 先切视频（若真的有视频）-------------
        if has_video:
            # 尝试 1：流拷贝（快）
            cmd_v_fast = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(start_time), "-t", str(duration), "-i", src_norm,
                "-map", "v:0", "-an",
                # 关键参数：尽量保证时间戳完整
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
                # 流拷贝失败，直接走重编码
                cmd_v_slow = [
                    "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                    "-i", src_norm, "-ss", str(start_time), "-t", str(duration),
                    "-map", "v:0", "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                    "-g", "16", "-keyint_min", "16", "-sc_threshold", "0",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart",
                    v_out
                ]
                subprocess.check_call(cmd_v_slow)
            else:
                # 流拷贝写成功，但不代表可读，验证一遍
                if not _verify_openable(v_out):
                    # 回退：删除坏文件，重切重编码
                    logger.warning('[FFmpeg] 流拷贝切片视频文件无法被OpenCV打开, 删除坏文件, 回落到重编码切分。')
                    try:
                        os.remove(v_out)
                    except Exception:
                        pass
                    cmd_v_slow = [
                        "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
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

        # ------------- 再切音频（若真的有音频）-------------
        if has_audio_real:
            cmd_a = [
                "ffmpeg", "-y", "-hide_banner", "-loglevel", "error",
                "-ss", str(start_time), "-t", str(duration), "-i", src_norm,
                "-map", "a:0", "-ac", "1", "-ar", "16000", "-c:a", "pcm_s16le",
                a_out
            ]
            subprocess.check_call(cmd_a)
            a_path_ret = a_out
        else:
            a_path_ret = None

        return {
            "video_path": v_path_ret,
            "audio_path": a_path_ret,
            "t0": start_time,
            "t1": start_time + duration,
            "duration": duration,
            "index": segment_index,
            "have_audio": bool(a_path_ret),  # 用“实际产出”作为 has_audio
        }


    # ========= 连续切窗生成器 =========
    @staticmethod
    def streaming_cut_generator(
        src_url: str, output_dir: str,
        slice_sec: int = 10, have_audio: Optional[bool] = None,  # 参数保留兼容，但不再依赖它判定
        start_offset: float = 0.0, max_duration: Optional[float] = None,
    ):
        """
        连续流式切窗（生成标准化后片段）：
        - 本地文件：探测总时长，到末尾停止
        - RTSP/直播：无限循环（外部 STOP 中断）
        - 片段内的音/视频轨道是否存在，以 cut_and_standardize_segment 的“实际产出”为准
        """
        FFmpegUtils.ensure_ffmpeg()
        os.makedirs(output_dir, exist_ok=True)

        # 离线文件：尽量探测总时长
        total_dur = FFmpegUtils._probe_duration_seconds(src_url)
        if total_dur is not None and max_duration is None:
            max_duration = total_dur

        seg_idx = 0
        t0 = float(start_offset)

        logger.info(f"开始流式切窗: {src_url}, 窗口={slice_sec}s")
        while True:
            if max_duration is not None and t0 >= max_duration:
                logger.info(f"[FFmpeg] 已到达文件末尾，总时长 {max_duration:.2f}s，结束切窗。")
                break

            # 这里不再依赖 have_audio；由下游函数自检轨道并按实际产出
            seg = FFmpegUtils.cut_and_standardize_segment(
                src_url=src_url, start_time=t0,
                duration=slice_sec, output_dir=output_dir,
                segment_index=seg_idx, have_audio=True  # 传什么都行，函数里已按实际轨道处理
            )

            # 仅用于更清晰的日志：回显“实际是否产出了音/视频”
            has_v = bool(seg.get("video_path"))
            has_a = bool(seg.get("audio_path"))
            logger.debug(
                "[FFmpeg] 切片完成 seg#%04d t0=%.3f t1=%.3f 产出: video=%s audio=%s",
                seg_idx, seg["t0"], seg["t1"],
                "yes" if has_v else "no",
                "yes" if has_a else "no",
            )

            yield seg

            seg_idx += 1
            t0 += slice_sec


    # 静音探测函数
    def detect_silence_intervals(
    src_url: str,
    noise_db: float = -35.0,          # 静音阈值（dB），数值越大越“敏感”，常用 -20 ~ -40dB
    min_silence: float = 0.5,         # 静音最短持续时间（秒）
    audio_stream_index: Optional[int] = None,  # 指定音频流（如 0 表示第1条音轨）；None=自动
    timeout_sec: Optional[float] = None,       # 最长执行时间；None=不限制（RTSP 建议传）
    max_intervals: Optional[int] = None,       # 最多返回多少段，None=不限制
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

        说明：
        - noise_db：阈值越接近 0 越容易判静音（例如 -25dB 比 -35dB 更严格）
        - min_silence：小于该时长的不算静音
        - 对 RTSP：建议配合 timeout_sec，超时将终止 ffmpeg 并返回已解析出的区间
        """

        def _normalize(p: str) -> str:
            return p.replace("file://", "", 1) if p.startswith("file://") else p

        src = _normalize(src_url)

        # 构造 ffmpeg 命令
        # -nostats/-hide_banner 精简输出；-vn 忽略视频；-f null - 丢弃输出
        # -map 选择音频流（如果指定了 audio_stream_index）
        filter_expr = f"silencedetect=noise={noise_db}dB:d={min_silence}"
        cmd = ["ffmpeg", "-hide_banner", "-nostats", "-v", "info"]

        # 对 RTSP 可按需增加：["-rtsp_transport", "tcp"]
        # 仅解码音频，降低开销
        cmd += ["-i", src]
        if audio_stream_index is not None:
            cmd += ["-map", f"0:a:{audio_stream_index}"]
        else:
            cmd += ["-map", "0:a:0"]  # 默认选第一条音轨；若无音轨会报错

        cmd += ["-af", filter_expr, "-vn", "-f", "null", "-"]

        # 启动子进程并逐行读 stderr
        proc = subprocess.Popen(
            cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True, bufsize=1
        )

        intervals: List[Dict[str, Optional[float]]] = []
        cur_start: Optional[float] = None
        start_ts = time.time()

        try:
            assert proc.stderr is not None
            for line in iter(proc.stderr.readline, ''):
                # 可选：超时控制（RTSP/直播强烈建议）
                if timeout_sec is not None and (time.time() - start_ts) > timeout_sec:
                    break

                # 解析 silence_start
                m1 = SILENCE_START_RE.search(line)
                if m1:
                    try:
                        cur_start = float(m1.group(1))
                    except Exception:
                        cur_start = None
                    continue

                # 解析 silence_end 与 duration
                m2 = SILENCE_END_RE.search(line)
                if m2:
                    try:
                        end = float(m2.group(1))
                        dur = float(m2.group(2))
                    except Exception:
                        end, dur = None, None

                    if cur_start is None and end is not None and dur is not None:
                        # 有些容器只打印 end/duration，也能反推 start
                        cur_start = max(0.0, end - dur)

                    intervals.append({"start": cur_start, "end": end, "duration": dur})
                    cur_start = None  # 归位

                    if max_intervals is not None and len(intervals) >= max_intervals:
                        break

            # 读完或超时后，尝试优雅结束
            try:
                proc.terminate()
            except Exception:
                pass
            proc.wait(timeout=2)
        except Exception:
            # 异常情况下也尽量终止子进程
            try:
                proc.kill()
            except Exception:
                pass
            raise
        finally:
            # 若在静音中提前结束（如 timeout/流中断），补上一条未闭合区间
            if cur_start is not None:
                intervals.append({"start": cur_start, "end": None, "duration": None})

        return intervals        
