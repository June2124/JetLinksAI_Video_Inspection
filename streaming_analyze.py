'''
Author: 13594053100@163.com
Date: 2025-10-17 15:21:53
LastEditTime: 2025-10-17 15:22:00
'''

from __future__ import annotations
# -*- coding: utf-8 -*-

import os
import queue
import threading
import time
from typing import List, Optional, Callable, Dict, Any, Iterator

from src.all_enum import MODEL, SOURCE_KIND
from src.utils import logger_utils
from src.utils.ffmpeg_utils import FFmpegUtils
from src.workers import worker_a_cut, worker_b_vlm, worker_c_asr

try:
    from src.utils.backpressure import AVBackpressureController
except Exception:
    AVBackpressureController = None  # type: ignore
try:
    from src.utils.skew_guard import TranscriptPlaybackSkewController
except Exception:
    TranscriptPlaybackSkewController = None  # type: ignore

logger = logger_utils.get_logger(__name__)


class StreamingAnalyze:
    
    """
    流式音视频分析主控（A/B/C 线程：切片→视觉解析→语音转写）

    线程拓扑  
        A：切片与标准化（输出：小视频/关键帧给 B；WAV 给 C）  
        B：视觉解析（流式增量 + 收尾）  
        C：ASR 转写（默认句级收尾；需要字级请改 C 线程策略）  
        OUT-*：内部消费者；统一做回调/打印；并把原始事件推送给 `run_stream()` 生成器

    用法（3选1）:
      1) run_stream(): 启动后台管线并返回一个“同步生成器”
         - VLM：持续按增量事件 yield（含原始 B 侧 payload）
         - ASR：持续按“句级收尾” yield（含原始 C 侧 payload）
      2) run_and_return(): 任务结束后一次性返回汇总（适合离线文件）
         - VLM：汇总 deltas/dones
         - ASR：汇总句级 dones（可在 C 侧改回字级）
      3) 自定义回调：设置 on_vlm_delta/on_vlm_done/on_asr_*，由内部 OUT-* 消费者实时触发

    重要参数:
      - url: 本地文件或 rtsp/rtsps
      - mode: MODEL.ONLINE / OFFLINE / SECURITY # 离线视频选择OFFLINE、实时流会议直播等场景选择ONLINE、实时流安防摄像头等场景选择SECURITY; 详见work_a_cut.py
      - slice_sec: A 侧切窗秒数
      - enable_b / enable_c: 是否启用 VLM / ASR 分支

    行为与约束:
      - 单实例不可并发运行多个任务；可串行多次调用
      - 离线：A 切到 EOF 后“慢停”，等 B/C 消费完自然退出
      - 实时：A 常驻；随时 force_stop("reason") 快停
      - 任一 B/C 异常退出 → 主控检测到后广播 STOP（快停）
      - 内置 TranscriptPlaybackSkewController：跨通道节流&对齐（可通过开关禁用/环境变量配置）

    事件时间戳:
      - 所有 run_stream 产出的事件都带 _meta.emit_ts / _meta.emit_iso
      - VLM 事件携带片段 t0/t1（由 A 侧提供）
      - ASR 事件为句级 done；含 t0/t1 与（如可用）句内时间戳列表

    常用环境变量  
        `EMIT_SKEW_GUARD`：是否启用对齐/节流（"1/true/on/yes" 开启；"0/false/off/no" 关闭；默认开启）  
        `EMIT_MAX_SKEW_S`, `EMIT_RATE_LIMIT_HZ` —— 对齐/节流参数  
        `VLM_MODEL_NAME`, `VLM_USER_PROMPT_*` —— VLM 配置  
        `ASR_MODEL`, `ASR_VAD_*` —— ASR/VAD 配置
    """

    def __init__(
        self,
        url: str,
        mode: MODEL,
        slice_sec: Optional[int] = None, # SECURITY模式建议 4~8s; ONLINE模式建议 5~10s；OFFLINE模式建议: 8~12s
        *,
        enable_b: bool = True,
        enable_c: bool = True,
        skew_guard_enabled: Optional[bool] = None,  # 对齐/节流总开关
    ):
        if not isinstance(mode, MODEL):
            raise ValueError(f"mode 只接受 MODEL 枚举，但传入了 {type(mode)}")
        
        _check_url_legal(url)
        FFmpegUtils.ensure_ffmpeg()

        # 按模式给默认窗口（秒）
        default_slice = {
            MODEL.ONLINE: 6,
            MODEL.SECURITY: 6,
            MODEL.OFFLINE: 10,
        }[mode]

        if slice_sec is None:
            slice_val = default_slice # 不显示传参, 则按照工作模式, 给出系统默认值
        else:
            if not isinstance(slice_sec, int) or not (0 < slice_sec < 30):
                raise ValueError("切窗参数必须是 (4, 30) 之间的正整数(单位s); 注意: slice_sec参数与首帧响应时间、CPU消耗正相关, Token消耗负相关。")
            slice_val = slice_sec 


        self.url = url
        self.mode = mode
        self.slice_sec = slice_val # 统一写回
        self._source_kind = _determine_source_kind(url)
        self._have_audio_track = FFmpegUtils.have_audio_track(url)

        self.enable_b = bool(enable_b)
        self.enable_c = bool(enable_c)

        self.on_vlm_delta: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_vlm_done: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_asr_no_speech: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_asr_delta: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_asr_done: Optional[Callable[[Dict[str, Any]], None]] = None

        self._run_lock = threading.Lock()
        self._running = False

        # -------- backpressure 可选 --------
        self.backpressure = None
        if AVBackpressureController:
            try:
                self.backpressure = AVBackpressureController(
                    max_q_video=80, max_window_delay_s=8, cooldown_s=0.2
                )
            except Exception:
                self.backpressure = None

        # -------- skew_guard 开关解析 --------
        def _str2bool(s: str) -> Optional[bool]:
            s = s.strip().lower()
            if s in ("1", "true", "on", "yes", "y"):  return True
            if s in ("0", "false", "off", "no", "n"): return False
            return None

        if skew_guard_enabled is None:
            env_flag = os.getenv("EMIT_SKEW_GUARD", "")
            val = _str2bool(env_flag) if env_flag else None
            self._skew_guard_enabled = True if val is None else bool(val)
        else:
            self._skew_guard_enabled = bool(skew_guard_enabled)

        # -------- skew_guard 实例化（或禁用）--------
        self.skew_guard = None
        if self._skew_guard_enabled and TranscriptPlaybackSkewController:
            try:
                self.skew_guard = TranscriptPlaybackSkewController(
                    max_visual_skew_s=float(os.getenv("EMIT_MAX_SKEW_S", "3.0")),
                    max_emit_rate_hz=float(os.getenv("EMIT_RATE_LIMIT_HZ", "8.0")),
                )
            except Exception:
                self.skew_guard = None

        self._init_runtime_state()

        logger.info(
            "初始化流式分析：mode=%s, url=%s, source=%s, has_audio=%s, slice_sec=%s, enable_b=%s, enable_c=%s, skew_guard=%s",
            self.mode.value, self.url, self._source_kind.value, self._have_audio_track, self.slice_sec,
            self.enable_b, self.enable_c, ("on" if self.skew_guard else "off")
        )

    # ---------- 运行期状态 ----------
    def _init_runtime_state(self):
        self._Q_VIDEO: queue.Queue = queue.Queue(maxsize=100)
        self._Q_AUDIO: queue.Queue = queue.Queue(maxsize=100)
        self._Q_VLM:   queue.Queue = queue.Queue(maxsize=500)
        self._Q_ASR:   queue.Queue = queue.Queue(maxsize=500)

        self._Q_CTRL_A: queue.Queue = queue.Queue(maxsize=50)
        self._Q_CTRL_B: queue.Queue = queue.Queue(maxsize=50)
        self._Q_CTRL_C: queue.Queue = queue.Queue(maxsize=50)

        # 对外事件总线（run_stream 用）
        self._Q_EVENTS: queue.Queue = queue.Queue(maxsize=2000)
        self._events_done = threading.Event()

        self._STOP = object()

        self._threads: List[threading.Thread] = []
        self._consumer_threads: List[threading.Thread] = []
        self._consumers_started = False

        self._monitor_thread: Optional[threading.Thread] = None
        self._monitor_stop = threading.Event()
        self._monitor_enabled = False

        self._last_vlm_emit_ts = 0.0
        self._last_asr_emit_ts = 0.0

        self._stats_lock = threading.Lock()
        self._stats = {
            "asr": {
                "no_speech_segments": 0,
                "no_speech_duration_s": 0.0,
                "segments": 0,
                "text_chars": 0,
            },
            "vlm": {
                "segments": 0,
                "deltas": 0,
                "text_chars": 0,
                "latency_ms_sum": 0.0,
                "latency_ms_max": 0.0,
            }
        }

        self._stopped = False
        self._stopped_once = False

    def _reset_runtime(self):
        self._init_runtime_state()

    # ---------- 对外：运行期切换 skew_guard ----------
    def set_skew_guard(self, enabled: bool) -> None:
        """
        运行期切换对齐/节流（skew_guard）。True 启用，False 关闭。
        """
        self._skew_guard_enabled = bool(enabled)
        if enabled and TranscriptPlaybackSkewController:
            try:
                self.skew_guard = TranscriptPlaybackSkewController(
                    max_visual_skew_s=float(os.getenv("EMIT_MAX_SKEW_S", "3.0")),
                    max_emit_rate_hz=float(os.getenv("EMIT_RATE_LIMIT_HZ", "8.0")),
                )
                logger.info("[主控] SkewGuard 已启用。")
            except Exception as e:
                self.skew_guard = None
                logger.warning("[主控] SkewGuard 启用失败，已降级关闭：%s", e)
        else:
            self.skew_guard = None
            logger.info("[主控] SkewGuard 已关闭。")

    # ---------- 对外入口 ----------
    def start_streaming_analyze(self):
        with self._run_lock:
            if self._running:
                logger.warning("[主控] 已在运行，忽略本次启动。")
                return
            self._running = True

        try:
            self._reset_runtime()
            self._start_monitor()
            self._start_output_consumers()

            if self._source_kind == SOURCE_KIND.AUDIO_FILE:
                logger.info("[主控] 离线音频流程启动")
                self._start_audio_file_streaming_analyze()
            elif self._source_kind == SOURCE_KIND.VIDEO_FILE:
                logger.info("[主控] 离线视频流程启动")
                self._start_video_file_streaming_analyze()
            else:
                logger.info("[主控] 实时流流程启动")
                self._start_real_time_streaming_analyze()
        except Exception as e:
            logger.exception("[主控] 运行异常：%s", e)
            raise
        finally:
            try:
                self._graceful_stop()
            except Exception as e:
                logger.exception("[主控] 收尾过程中出现异常（已忽略以保证状态复位）：%s", e)
            finally:
                with self._run_lock:
                    self._running = False

    # ---------- 分支控制 ----------
    def _watch_branch(
        self,
        t_a: threading.Thread,
        t_b: Optional[threading.Thread] = None,
        t_c: Optional[threading.Thread] = None,
        *,
        need_video_sentinel: bool,
        need_audio_sentinel: bool,
        poll_sec: float = 0.5,
    ):
        while t_a.is_alive():
            if (t_b and not t_b.is_alive()) or (t_c and not t_c.is_alive()):
                logger.error("[主控] 下游线程异常退出（B/C），广播 STOP 强制停止全部线程。")
                self._broadcast_ctrl({"type": "STOP", "reason": "B/C thread died"})
                _safe_put(self._Q_VIDEO, self._STOP)
                _safe_put(self._Q_AUDIO, self._STOP)
                return
            t_a.join(timeout=poll_sec)

        if need_video_sentinel:
            _safe_put(self._Q_VIDEO, self._STOP)
        if need_audio_sentinel:
            _safe_put(self._Q_AUDIO, self._STOP)

        self._wait_workers_quietly(*(t for t in (t_b, t_c) if t), poll_sec=poll_sec)

    def _wait_workers_quietly(self, *workers: threading.Thread, poll_sec: float = 0.5):
        while True:
            if self._stopped:
                logger.info("[主控] 强停标记已设置，停止等待下游线程")
                return
            alive = [t.name for t in workers if t and t.is_alive()]
            if not alive:
                logger.info('[主控] 下游线程已全部退出')
                return
            logger.debug("[主控] 等待下游线程自然退出：%s", alive)
            time.sleep(poll_sec)

    # ---------- 启动各模式 ----------
    def _start_audio_file_streaming_analyze(self):
        t_a, t_b, t_c = self._spawn_threads()

        to_start = [t_a]
        if self.enable_b and t_b:
            to_start.append(t_b)
        if self.enable_c and t_c:
            to_start.append(t_c)

        self._start_threads(*to_start)

        self._broadcast_ctrl({"type": "START"})
        self._broadcast_ctrl({"type": "MODE_CHANGE", "value": self.mode.value})
        self._broadcast_ctrl({"type": "UPDATE_SLICE", "value": self.slice_sec})

        self._watch_branch(
            t_a,
            t_b=(t_b if self.enable_b else None),
            t_c=(t_c if self.enable_c else None),
            need_video_sentinel=self.enable_b,
            need_audio_sentinel=self.enable_c,
        )

    def _start_video_file_streaming_analyze(self):
        t_a, t_b, t_c = self._spawn_threads()

        to_start = [t_a]
        if self.enable_b and t_b:
            to_start.append(t_b)
        if self._have_audio_track and self.enable_c and t_c:
            to_start.append(t_c)

        self._start_threads(*to_start)

        self._broadcast_ctrl({"type": "START"})
        self._broadcast_ctrl({"type": "MODE_CHANGE", "value": self.mode.value})
        self._broadcast_ctrl({"type": "UPDATE_SLICE", "value": self.slice_sec})

        self._watch_branch(
            t_a,
            t_b=(t_b if self.enable_b else None),
            t_c=(t_c if (self._have_audio_track and self.enable_c) else None),
            need_video_sentinel=self.enable_b,
            need_audio_sentinel=(self._have_audio_track and self.enable_c),
        )

    def _start_real_time_streaming_analyze(self):
        t_a, t_b, t_c = self._spawn_threads()

        to_start = [t_a]
        if self.enable_b and t_b:
            to_start.append(t_b)
        if self._have_audio_track and self.enable_c and t_c:
            to_start.append(t_c)

        self._start_threads(*to_start)

        self._broadcast_ctrl({"type": "START"})
        self._broadcast_ctrl({"type": "MODE_CHANGE", "value": self.mode.value})
        self._broadcast_ctrl({"type": "UPDATE_SLICE", "value": self.slice_sec})

        self._watch_branch(
            t_a,
            t_b=(t_b if self.enable_b else None),
            t_c=(t_c if (self._have_audio_track and self.enable_c) else None),
            need_video_sentinel=self.enable_b,
            need_audio_sentinel=(self._have_audio_track and self.enable_c),
        )

    # ---------- 外部强停 ----------
    def force_stop(self, reason: Optional[str] = "无"):
        if getattr(self, "_stopped", False):
            logger.info("[主控] force_stop() 已调用过，本次忽略。")
            return
        self._stopped = True

        logger.info(f"[主控] 外部调用者强制中断，原因：{reason}")

        try:
            self._broadcast_ctrl({"type": "STOP", "reason": reason})
        except Exception as e:
            logger.warning(f"[主控] 向控制队列广播 STOP 失败：{e}")

        _safe_put(self._Q_VIDEO, self._STOP)
        _safe_put(self._Q_AUDIO, self._STOP)

        self._graceful_stop()
        logger.info("[主控] 强制退出完成！")

    # ---------- 线程管理 ----------
    def _spawn_threads(self):
        have_audio_for_a = bool(self._have_audio_track and self.enable_c)

        t_a = threading.Thread(
            target=worker_a_cut.worker_a_cut, daemon=True,
            args=(
                self.url,
                have_audio_for_a,
                self.mode,
                self.slice_sec,
                (self._Q_AUDIO if self.enable_c else None),
                (self._Q_VIDEO if self.enable_b else None),
                self._Q_CTRL_A,
                self._STOP,
            ),
            name="A-切片标准化"
        )

        t_b = None
        if self.enable_b:
            t_b = threading.Thread(
                target=worker_b_vlm.worker_b_vlm, daemon=True,
                args=(self._Q_VIDEO, self._Q_VLM, self._Q_CTRL_B, self._STOP,),
                name="B-VLM解析"
            )

        t_c = None
        if self.enable_c:
            t_c = threading.Thread(
                target=worker_c_asr.worker_c_asr, daemon=True,
                args=(self._Q_AUDIO, self._Q_ASR, self._Q_CTRL_C, self._STOP,),
                name="C-ASR转写"
            )

        self._threads = [t for t in (t_a, t_b, t_c) if t]
        logger.debug("[主控] 线程已创建：%s", [t.name for t in self._threads])
        return t_a, t_b, t_c

    def _start_threads(self, *threads: threading.Thread):
        started: List[threading.Thread] = []
        try:
            for t in threads:
                t.start()
                started.append(t)
                logger.info("[主控] 线程启动：%s (ident=%s)", t.name, t.ident)
        except Exception as e:
            logger.exception("[主控] 线程启动失败，触发快停兜底：%s", e)
            try:
                self._broadcast_ctrl({"type": "STOP", "reason": "startup failure"})
            except Exception as be:
                logger.debug("[主控] 启动失败时广播 STOP 异常（忽略）：%s", be)
            _safe_put(self._Q_VIDEO, self._STOP)
            _safe_put(self._Q_AUDIO, self._STOP)
            self._wait_workers_quietly(*started, poll_sec=0.2)
            raise

    # ---------- 控制广播 ----------
    def _broadcast_ctrl(self, msg: Dict[str, Any]):
        for q in (self._Q_CTRL_A, (self._Q_CTRL_B if self.enable_b else None), (self._Q_CTRL_C if self.enable_c else None)):
            if q is None:
                continue
            try:
                q.put_nowait(msg)
            except queue.Full:
                q.put(msg)

    # ---------- 输出消费者 ----------
    def _emit_to_event_bus(self, ev: Dict[str, Any], *, channel: str):
        try:
            now = time.time()
            out = dict(ev)
            meta = dict(out.get("_meta") or {})
            meta.update({
                "emit_ts": now,
                "emit_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
                "channel": channel,
            })
            out["_meta"] = meta
            self._Q_EVENTS.put_nowait(out)
        except queue.Full:
            logger.warning("[主控] 对外事件总线拥堵，丢弃 1 条：%s", ev.get("type"))
        except Exception:
            pass

    def _start_output_consumers(self):
        if self._consumers_started:
            return
        self._consumers_started = True

        try:
            if self.enable_b:
                t_v = threading.Thread(target=self._consume_vlm, name="OUT-VLM", daemon=True)
                self._consumer_threads.append(t_v)
            if self.enable_c:
                t_a = threading.Thread(target=self._consume_asr, name="OUT-ASR", daemon=True)
                self._consumer_threads.append(t_a)

            for t in self._consumer_threads:
                t.start()
                logger.info("[主控] 输出消费者启动：%s", t.name)
        except Exception as e:
            logger.exception("[主控] 输出消费者启动失败，触发快停兜底：%s", e)
            try:
                self._broadcast_ctrl({"type": "STOP", "reason": "consumer startup failure"})
            except Exception:
                pass
            _safe_put(self._Q_VIDEO, self._STOP)
            _safe_put(self._Q_AUDIO, self._STOP)
            self._wait_workers_quietly(*(t for t in self._consumer_threads if t), poll_sec=0.2)
            raise

    def _consume_vlm(self):
        while True:
            try:
                item = self._Q_VLM.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is self._STOP:
                logger.info("[OUT-VLM] 收到数据队列STOP哨兵，退出。")
                return

            now = time.time()
            allow_emit = True
            try:
                if self.skew_guard:
                    allow_emit = self.skew_guard.allow_emit_vlm(now, last_asr_ts=self._last_asr_emit_ts)
            except Exception:
                pass

            if item.get("type") == "vlm_stream_delta":
                if allow_emit:
                    self._emit_vlm_delta(item)
                    with self._stats_lock:
                        self._stats["vlm"]["deltas"] += 1
                    self._last_vlm_emit_ts = now
                    # VLM: 增量也发到对外事件总线（原样）
                    self._emit_to_event_bus(item, channel="vlm")

            elif item.get("type") == "vlm_stream_done":
                self._emit_vlm_done(item)
                text_len = len(item.get("full_text") or "")
                try:
                    lat = float(item.get("latency_ms") or 0.0)
                except Exception:
                    lat = 0.0
                with self._stats_lock:
                    self._stats["vlm"]["segments"] += 1
                    self._stats["vlm"]["text_chars"] += text_len
                    self._stats["vlm"]["latency_ms_sum"] += lat
                    if lat > self._stats["vlm"]["latency_ms_max"]:
                        self._stats["vlm"]["latency_ms_max"] = lat
                self._last_vlm_emit_ts = now
                # VLM: done 也发到对外事件总线（原样）
                self._emit_to_event_bus(item, channel="vlm")

            else:
                logger.debug("[OUT-VLM] 忽略未知消息：%s", item)

    def _consume_asr(self):
        while True:
            try:
                item = self._Q_ASR.get(timeout=0.1)
            except queue.Empty:
                continue
            if item is self._STOP:
                logger.info("[OUT-ASR] 收到数据队列STOP哨兵，退出。")
                return

            now = time.time()
            allow_emit = True
            try:
                if self.skew_guard:
                    allow_emit = self.skew_guard.allow_emit_asr(now, last_vlm_ts=self._last_vlm_emit_ts)
            except Exception:
                pass

            if item.get("type") == "asr_stream_delta":
                if allow_emit:
                    self._emit_asr_delta(item)
                    with self._stats_lock:
                        self._stats["asr"]["text_chars"] += len(item.get("delta") or "")
                    self._last_asr_emit_ts = now
                # 不把 delta 投到对外事件总线（run_stream 对 ASR 只输出 done）

            elif item.get("type") == "asr_stream_done":
                self._emit_asr_done(item)
                with self._stats_lock:
                    self._stats["asr"]["segments"] += 1
                    self._stats["asr"]["text_chars"] += len(item.get("full_text") or "")
                self._last_asr_emit_ts = now
                # 仅把 done 投到对外事件总线
                self._emit_to_event_bus(item, channel="asr")

            elif item.get("type") == "asr_stream_no_speech":
                self._emit_asr_no_speech(item)
                self._last_asr_emit_ts = now
                # no_speech 也不投到对外事件总线

            else:
                logger.debug("[OUT-ASR] 忽略未知消息：%s", item)

    # ---------- 发射 ----------
    def _emit_vlm_delta(self, payload: Dict[str, Any]):
        if callable(self.on_vlm_delta):
            try:
                self.on_vlm_delta(payload)
                return
            except Exception as e:
                logger.warning("[OUT] on_vlm_delta 回调异常：%s", e)
        logger.info(
            "[📺VLM增量 seg#%s seq=%s] %s",
            payload.get("segment_index"), payload.get("seq"),
            (payload.get("delta") or "").strip()
        )

    def _emit_vlm_done(self, payload: Dict[str, Any]):
        """
        VLM 段尾：当 suppressed_dup=True 且 full_text 为空，默认静默不打印；
        如需打印提示，请设置环境变量 VLM_LOG_SUPPRESS_EMPTY=0。
        事件始终会继续向外发送（run_stream 仍能拿到），不影响统计与回调。
        """
        # 回调优先（让外部有机会拿到原始 payload 作自定义处理/统计）
        if callable(self.on_vlm_done):
            try:
                self.on_vlm_done(payload)
                # 即使回调成功，也继续做下面的“可选日志打印”逻辑（保持与增量一致）
            except Exception as e:
                logger.warning("[OUT] on_vlm_done 回调异常：%s", e)

        text = (payload.get("full_text") or "").strip()
        suppressed = bool(payload.get("suppressed_dup"))
        suppress_empty_log = os.getenv("VLM_LOG_SUPPRESS_EMPTY", "1") == "1"

        # 1) 无新增且为空文本 -> 默认静默；如需提示，切换 env
        if suppressed and not text:
            if suppress_empty_log:
                return
            else:
                logger.info(
                    "[✨✨✨VLM无新增 seg#%s kind=%s ms=%s] (与历史一致，已省略)",
                    payload.get("segment_index"),
                    payload.get("media_kind"),
                    payload.get("latency_ms"),
                )
                return

        # 2) 正常打印完整文本
        logger.info(
            "[✨✨✨VLM完整文本 seg#%s kind=%s ms=%s]%s%s",
            payload.get("segment_index"),
            payload.get("media_kind"),
            payload.get("latency_ms"),
            (" [仅新增]" if suppressed is False else ""),  # suppressed=False 不一定等同“仅新增”，这里只是标个位
            ("\n" + text) if text else ""
        )


    def _emit_asr_no_speech(self, payload: Dict[str, Any]):
        t0 = payload.get("t0", 0.0)
        t1 = payload.get("t1", 0.0)
        try:
            dur = float(t1) - float(t0)
        except Exception:
            dur = 0.0
        self._stats_add_no_speech(dur)

        if callable(self.on_asr_no_speech):
            try:
                self.on_asr_no_speech(payload)
                return
            except Exception as e:
                logger.warning("[OUT] on_asr_no_speech 回调异常：%s", e)

        usage = payload.get("usage") or {}
        vad_info = usage.get("vad") or {}
        backend = vad_info.get("backend_used", "unknown")
        s_hint = usage.get("silence_hint") or {}
        s_ratio = s_hint.get("silence_ratio")
        active = vad_info.get("active_ratio", 0.0)

        logger.info(
            "[❗ASR无人声跳过 seg#%s dur=%.3fs backend=%s silent_ratio=%s active_ratio=%s] %s",
            payload.get("segment_index"),
            max(0.0, dur),
            backend,
            (f"{s_ratio:.2f}" if isinstance(s_ratio, (int, float)) else "n/a"),
            (f"{active:.3f}" if isinstance(active, (int, float)) else "n/a"),
            (payload.get("full_text") or "").strip()
        )

    def _emit_asr_delta(self, payload: Dict[str, Any]):
        if callable(self.on_asr_delta):
            try:
                self.on_asr_delta(payload)
                return
            except Exception as e:
                logger.warning("[OUT] on_asr_delta 回调异常：%s", e)
        logger.info(
            "[🎵ASR增量 seg#%s seq=%s] %s",
            payload.get("segment_index"), payload.get("seq"),
            (payload.get("delta") or "").strip()
        )

    def _emit_asr_done(self, payload: Dict[str, Any]):
        if callable(self.on_asr_done):
            try:
                self.on_asr_done(payload)
                return
            except Exception as e:
                logger.warning("[OUT] on_asr_done 回调异常：%s", e)
        logger.info(
            "[🎉🎉🎉ASR完整文本 seg#%s] %s",
            payload.get("segment_index"),
            (payload.get("full_text") or "").strip()
        )

    # ---------- 优雅停止 ----------
    def _graceful_stop(self):
        if self._stopped_once:
            logger.warning('[主控] 调用链上游已触发优雅清理，本次调用跳过')
            return
        self._stopped_once = True

        need_stop = any(t and t.is_alive() for t in self._threads)
        if need_stop:
            try:
                self._broadcast_ctrl({"type": "STOP"})
            except Exception as e:
                logger.debug("[主控] 广播 STOP 控制失败：%s", e)

        try:
            self._stop_monitor()
        except Exception:
            pass

        for t in self._threads:
            try:
                if t and t.is_alive():
                    t.join(timeout=5.0)
            except Exception:
                pass

        if self._consumers_started:
            if self.enable_b:
                _safe_put(self._Q_VLM, self._STOP)
            if self.enable_c:
                _safe_put(self._Q_ASR, self._STOP)

        for t in self._consumer_threads:
            try:
                if t and t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                pass

        try:
            stats = self.snapshot_stats()
            if "vlm" in stats:
                vlm = stats["vlm"]
                avg_lat = (vlm["latency_ms_sum"] / vlm["segments"]) if vlm["segments"] else 0.0
                logger.info(
                    "[主控] VLM统计：segments=%d, deltas=%d, text_chars=%d, latency_avg=%.1fms, latency_max=%.1fms",
                    vlm["segments"], vlm["deltas"], vlm["text_chars"], avg_lat, vlm["latency_ms_max"]
                )
            if "asr" in stats:
                asr = stats["asr"]
                logger.info(
                    "[主控] ASR统计：segments=%d, text_chars=%d, no_speech_segments=%d, no_speech_duration=%.2fs",
                    asr["segments"], asr["text_chars"], asr["no_speech_segments"], asr["no_speech_duration_s"]
                )
        except Exception:
            pass

        # 通知 run_stream 退出
        self._events_done.set()
        logger.info("[主控] 全部线程结束或已交由进程回收")

    # ---------- 监控 ----------
    def _start_monitor(self, interval: float = 2.0):
        if self._monitor_thread and self._monitor_thread.is_alive():
            return

        def _loop():
            logger.info("[监控] 启动，间隔 %.1fs", interval)
            while not self._monitor_stop.is_set():
                try:
                    qv   = self._Q_VIDEO.qsize() if self.enable_b else None
                    qa   = self._Q_AUDIO.qsize() if self.enable_c else None
                    qvlm = self._Q_VLM.qsize()   if self.enable_b else None
                    qasr = self._Q_ASR.qsize()   if self.enable_c else None
                    cA = self._Q_CTRL_A.qsize()
                    cB = self._Q_CTRL_B.qsize() if self.enable_b else None
                    cC = self._Q_CTRL_C.qsize() if self.enable_c else None
                    alive = {t.name: t.is_alive() for t in self._threads if t}

                    def f(v): return "-" if v is None else str(v)
                    logger.info(
                        "[监控] 队列水位 VIDEO=%s AUDIO=%s VLM=%s ASR=%s | CTRL_A=%s CTRL_B=%s CTRL_C=%s | 线程存活=%s",
                        f(qv), f(qa), f(qvlm), f(qasr), cA, f(cB), f(cC), alive
                    )
                except Exception as e:
                    logger.debug("[监控] 采集异常：%s", e)
                finally:
                    time.sleep(interval)
            logger.info("[监控] 已停止")

        self._monitor_stop.clear()
        try:
            self._monitor_thread = threading.Thread(target=_loop, name="Monitor-监控", daemon=True)
            self._monitor_thread.start()
            self._monitor_enabled = True
        except Exception as e:
            self._monitor_enabled = False
            self._monitor_thread = None
            logger.warning("[监控] 启动失败，进入无监控降级模式：%s", e)

    def _stop_monitor(self, join_timeout: float = 2.0):
        self._monitor_stop.set()
        t = self._monitor_thread
        if t and t.is_alive():
            try:
                t.join(timeout=join_timeout)
            except Exception as e:
                logger.debug("[监控] 停止时异常（忽略）：%s", e)
        self._monitor_enabled = False

    # ---------- 统计 ----------
    def _stats_add_no_speech(self, duration_s: float):
        if not self.enable_c:
            return
        if duration_s < 0:
            duration_s = 0.0
        with self._stats_lock:
            self._stats["asr"]["no_speech_segments"] += 1
            self._stats["asr"]["no_speech_duration_s"] += float(duration_s)

    def snapshot_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            out: Dict[str, Any] = {}
            if self.enable_b:
                out["vlm"] = dict(self._stats["vlm"])
            if self.enable_c:
                out["asr"] = dict(self._stats["asr"])
            return out

    def reset_stats(self):
        with self._stats_lock:
            if self.enable_c:
                self._stats["asr"]["no_speech_segments"] = 0
                self._stats["asr"]["no_speech_duration_s"] = 0.0
                self._stats["asr"]["segments"] = 0
                self._stats["asr"]["text_chars"] = 0
            if self.enable_b:
                self._stats["vlm"]["segments"] = 0
                self._stats["vlm"]["deltas"] = 0
                self._stats["vlm"]["text_chars"] = 0
                self._stats["vlm"]["latency_ms_sum"] = 0.0
                self._stats["vlm"]["latency_ms_max"] = 0.0

    # ---------- 便捷：打印到标准输出 ----------
    def set_stdout_handlers(self, *, print_vlm: bool = True, print_asr: bool = True) -> None:
        if print_vlm and self.enable_b:
            def _vlm_delta(p):
                seg, seq = p.get("segment_index"), p.get("seq")
                delta = (p.get("delta") or "").strip()
                if delta:
                    print(f"[VLMΔ seg#{seg} seq={seq}] {delta}", flush=True)

            def _vlm_done(p):
                seg = p.get("segment_index")
                text = (p.get("full_text") or "").strip()
                print(f"[VLM✓ seg#{seg}] {text}", flush=True)

            self.on_vlm_delta = _vlm_delta
            self.on_vlm_done = _vlm_done

        if print_asr and self.enable_c:
            def _asr_delta(p):
                seg, seq = p.get("segment_index"), p.get("seq")
                delta = (p.get("delta") or "").strip()
                if delta:
                    print(f"[ASRΔ seg#{seg} seq={seq}] {delta}", flush=True)

            def _asr_done(p):
                seg = p.get("segment_index")
                text = (p.get("full_text") or "").strip()
                print(f"[ASR✓ seg#{seg}] {text}", flush=True)

            def _asr_no_speech(p):
                seg = p.get("segment_index")
                hint = (p.get("usage") or {}).get("silence_hint") or {}
                ratio = hint.get("silence_ratio")
                print(f"[ASR⊘ seg#{seg}] silence_ratio={ratio}", flush=True)

            self.on_asr_delta = _asr_delta
            self.on_asr_done = _asr_done
            self.on_asr_no_speech = _asr_no_speech

    def run_simple(self, *, print_vlm: bool = True, print_asr: bool = True, max_secs: float | None = None) -> None:
        self.set_stdout_handlers(print_vlm=print_vlm, print_asr=print_asr)

        stop_flag = threading.Event()

        def _watchdog():
            if max_secs and max_secs > 0:
                t0 = time.time()
                while not stop_flag.is_set():
                    if time.time() - t0 >= max_secs:
                        try:
                            self.force_stop(f"timeout {max_secs}s")
                        except Exception:
                            pass
                        break
                    time.sleep(0.2)

        if max_secs and max_secs > 0:
            threading.Thread(target=_watchdog, daemon=True).start()

        try:
            self.start_streaming_analyze()
        finally:
            stop_flag.set()

    # ---------- 便捷：采集并一次性返回 ----------
    def set_capture_handlers(self, *, capture_vlm: bool = True, capture_asr: bool = True):
        from threading import RLock
        self._capture_lock = getattr(self, "_capture_lock", RLock())
        self._capture_buf  = getattr(self, "_capture_buf", {"vlm": {"deltas": [], "dones": []},
                                                            "asr": {"deltas": [], "dones": [], "no_speech": []}})

        if capture_vlm and self.enable_b:
            def _vlm_delta(p):
                with self._capture_lock:
                    self._capture_buf["vlm"]["deltas"].append(dict(p))
            def _vlm_done(p):
                with self._capture_lock:
                    self._capture_buf["vlm"]["dones"].append(dict(p))
            self.on_vlm_delta = _vlm_delta
            self.on_vlm_done  = _vlm_done

        if capture_asr and self.enable_c:
            def _asr_delta(p):
                with self._capture_lock:
                    self._capture_buf["asr"]["deltas"].append(dict(p))
            def _asr_done(p):
                with self._capture_lock:
                    self._capture_buf["asr"]["dones"].append(dict(p))
            def _asr_no_speech(p):
                with self._capture_lock:
                    self._capture_buf["asr"]["no_speech"].append(dict(p))
            self.on_asr_delta     = _asr_delta
            self.on_asr_done      = _asr_done
            self.on_asr_no_speech = _asr_no_speech

    def run_and_return(self, *, print_vlm: bool = False, print_asr: bool = False,
                       max_secs: float | None = None) -> dict:
        self.set_capture_handlers(capture_vlm=True, capture_asr=True)
        if print_vlm or print_asr:
            self.set_stdout_handlers(print_vlm=print_vlm, print_asr=print_asr)

        stop_flag = threading.Event()

        def _watchdog():
            if max_secs and max_secs > 0:
                t0 = time.time()
                while not stop_flag.is_set():
                    if time.time() - t0 >= max_secs:
                        try:
                            self.force_stop(f"timeout {max_secs}s")
                        except Exception:
                            pass
                        break
                    time.sleep(0.2)

        if max_secs and max_secs > 0:
            threading.Thread(target=_watchdog, daemon=True).start()

        try:
            self.start_streaming_analyze()
        finally:
            stop_flag.set()

        out = {}
        with getattr(self, "_capture_lock"):
            if self.enable_b:
                out["vlm"] = {
                    "deltas": list(self._capture_buf["vlm"]["deltas"]),
                    "dones":  list(self._capture_buf["vlm"]["dones"]),
                }
            if self.enable_c:
                out["asr"] = {
                    "deltas":     list(self._capture_buf["asr"]["deltas"]),
                    "dones":      list(self._capture_buf["asr"]["dones"]),
                    "no_speech":  list(self._capture_buf["asr"]["no_speech"]),
                }
        return out

    # ======================== 原始流式输出生成器 ========================
    def run_stream(self, *, print_vlm: bool = False, print_asr: bool = False,
                   max_secs: float | None = None) -> Iterator[Dict[str, Any]]:
        """
        启动整条管线到后台线程，并立即返回一个同步生成器。
        迭代器会持续 yield 事件（VLM: 增量+收尾；ASR: 仅收尾），直到任务优雅收尾。
        """
        if print_vlm or print_asr:
            self.set_stdout_handlers(print_vlm=print_vlm, print_asr=print_asr)

        def _runner():
            try:
                self.start_streaming_analyze()
            except Exception:
                logger.exception("[主控] 后台运行异常")

        t = threading.Thread(target=_runner, name="Runner-StreamingAnalyze", daemon=True)
        t.start()

        stop_flag = threading.Event()

        def _watchdog():
            if max_secs and max_secs > 0:
                t0 = time.time()
                while not stop_flag.is_set():
                    if time.time() - t0 >= max_secs:
                        try:
                            self.force_stop(f"timeout {max_secs}s")
                        except Exception:
                            pass
                        break
                    time.sleep(0.2)

        if max_secs and max_secs > 0:
            threading.Thread(target=_watchdog, daemon=True).start()

        try:
            while True:
                try:
                    ev = self._Q_EVENTS.get(timeout=0.2)
                    yield ev
                except queue.Empty:
                    if self._events_done.is_set():
                        try:
                            while True:
                                ev = self._Q_EVENTS.get_nowait()
                                yield ev
                        except queue.Empty:
                            break
                        finally:
                            break
                    continue
        finally:
            stop_flag.set()
            try:
                t.join(timeout=2.0)
            except Exception:
                pass


# ----------------- 工具函数 -----------------
def _safe_put(q: queue.Queue, item: Any, *, timeout: float = 0.2):
    try:
        q.put(item, timeout=timeout)
    except Exception as e:
        logger.debug("[主控] put 响应异常（忽略）：%s", e)


def _check_url_legal(url: str) -> None:
    if not url or not isinstance(url, str):
        raise ValueError("url 不能为空且必须是字符串类型")
    if url.startswith(("rtsp://", "rtsps://")):
        return
    if url.startswith("file://"):
        local_path = url.replace("file://", "", 1)
        if not os.path.exists(local_path):
            raise ValueError(f"本地文件路径不存在: {local_path}")
        return
    if os.path.exists(url):
        return
    raise ValueError(f"不支持的媒体源地址格式: {url}，仅支持本地文件路径或 RTSP 流")


def _determine_source_kind(url: str) -> SOURCE_KIND:
    if url.startswith(("rtsp://", "rtsps://")):
        return SOURCE_KIND.RTSP
    elif url.startswith("file://") or os.path.exists(url):
        video_extensions = [".mp4", ".avi", ".mkv", ".mov", ".flv"]
        audio_extensions = [".mp3", ".wav", ".aac", ".flac", ".ogg"]
        path = url.replace("file://", "", 1) if url.startswith("file://") else url
        ext = os.path.splitext(path)[1].lower()
        if ext in video_extensions:
            return SOURCE_KIND.VIDEO_FILE
        elif ext in audio_extensions:
            return SOURCE_KIND.AUDIO_FILE
        else:
            raise ValueError(
                f"不支持的本地文件格式: {ext}，仅支持音频{audio_extensions}和视频{video_extensions}"
            )
    else:
        raise ValueError(f"无法确定媒体源类型，URL 格式不支持: {url}")
