from __future__ import annotations
'''
Author: 13594053100@163.com
Date: 2025-10-17 15:21:53
LastEditTime: 2025-11-19 18:43:40
'''

import os
import queue
import threading
import time
from typing import List, Optional, Callable, Dict, Any, Iterator
from pathlib import Path

from src.all_enum import MODEL, SOURCE_KIND
from src.utils import logger_utils
from src.workers import worker_a_cut, worker_b_vlm, worker_c_asr
from src.configs.vlm_config import VlmConfig
from src.configs.asr_config import AsrConfig
from src.configs.cut_config import CutConfig
from src.configs.rtsp_batch_config import RTSPBatchConfig, RTSP
from src.configs.runtime_machine_config import RuntimeMachineConfig
from src.runtime_machine.local_vlm_runtime_machine import LocalVlmRuntimeMachine
from src.runtime_machine.queue_runtime_machine import QueueRuntimeMachine
from JetLinksAI.src.utils.ffmpeg.python_ffmpeg_utils import (
    ensure_ffmpeg,
    have_audio_track,
)

logger = logger_utils.get_logger(__name__)


class StreamingAnalyze:
    # -------- 全局文件清理守护线程（仅启动一次） --------
    _cleanup_daemon_started: bool = False
    _cleanup_daemon_lock = threading.Lock()

    @classmethod
    def _start_cleanup_daemon_if_needed(cls) -> None:
        """
        启动文件清理守护线程（全局只会启动一次）：
        - A 侧切片目录：<项目根>/static/out
        - 证据帧目录：<项目根>/static/evidence_images
        """
        with cls._cleanup_daemon_lock:
            if cls._cleanup_daemon_started:
                return

            
            base_path = Path(__file__).resolve().parent
            static_root = base_path / "static"

            out_dir = static_root / "out"                 # /static/out
            evidence_dir = static_root / "evidence_images"  # /static/evidence_images

            try:
                # 延迟导入，避免循环依赖
                from src.utils.file_cleanup import start_cleanup_daemon

                start_cleanup_daemon(
                    out_dir=out_dir,
                    evidence_dir=evidence_dir,
                    out_ttl_hours=2,     # out 下原始流切片保留 24 小时
                    evidence_ttl_days=1,  # 证据帧保留 7 天
                    interval_hours=1,     # 每 1 小时跑一次清理
                )
                logger.info(
                    "[主控] 文件清理守护线程已启动: out_dir=%s, evidence_dir=%s",
                    out_dir, evidence_dir,
                )
            except Exception as e:
                # 不影响主流程，只做告警
                logger.warning("[主控] 文件清理守护线程启动失败（已忽略，不影响主流程）：%s", e)

            # 标记为已尝试启动（无论成功/失败，避免每次 new 都重复尝试）
            cls._cleanup_daemon_started = True

    
    def __init__(
        self,
        mode: MODEL,
        url: Optional[str] = None,  # 只有 OFFLINE 才能传入此参数
        *,
        enable_b: bool = True,
        enable_c: bool = True,
        rtsp_batch_config: Optional[RTSPBatchConfig] = None,  # SECURITY_SINGLE / SECURITY_POLLING 才能传入
        vlm_config: Optional[VlmConfig] = None,
        asr_config: Optional[AsrConfig] = None,
        cut_config: Optional[CutConfig] = None,
        runtime_machine_config: RuntimeMachineConfig = RuntimeMachineConfig()
    ):
        if not isinstance(mode, MODEL):
            raise ValueError(f"mode 只接受 MODEL 枚举，但传入了 {type(mode)}")

        # ffmpeg 环境检查
        ensure_ffmpeg()

         # 启动全局文件清理守护线程（仅首次有效）
        self._start_cleanup_daemon_if_needed()

        self.mode = mode
        self.enable_b = bool(enable_b)
        self.enable_c = bool(enable_c)
        self.rtsp_batch_config = rtsp_batch_config
        self.runtime_machine_config = runtime_machine_config

        # ---------- 其它配置默认化（必须在后面使用前完成） ----------
        self.vlm_config = vlm_config or VlmConfig()
        self.asr_config = asr_config or AsrConfig()
        self.cut_config = cut_config or CutConfig()

        # 运行时机器占位
        self._vlm_runtime_machine: Optional[LocalVlmRuntimeMachine] = None
        self._queue_runtime_machine: Optional[QueueRuntimeMachine] = None

        # ========= 1. SECURITY_SINGLE 和 SECURITY_POLLING =========
        if self.mode in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING):
            if url:
                raise ValueError("SECURITY_SINGLE/SECURITY_POLLING 不能使用 url 传递地址")

            if self.rtsp_batch_config is None or len(self.rtsp_batch_config.polling_list) == 0:
                raise ValueError("SECURITY_SINGLE/SECURITY_POLLING 模式下必须传入 rtsp_batch_config 且 polling_list 非空")

            if self.mode == MODEL.SECURITY_SINGLE and len(self.rtsp_batch_config.polling_list) > 1:
                self.rtsp_batch_config.polling_list = self.rtsp_batch_config.polling_list[:1]
                logger.warning("SECURITY_SINGLE 下只允许 polling_list 长度为 1，已强制只取第一个流地址")

            if self.mode == MODEL.SECURITY_POLLING and len(self.rtsp_batch_config.polling_list) < 2:
                raise ValueError("SECURITY_POLLING 下 polling_list 的长度必须至少为 2")
            if self.mode == MODEL.SECURITY_POLLING and len(self.rtsp_batch_config.polling_list) > 50:
                raise ValueError("SECURITY_POLLING 下 polling_list 的长度最多为 50")

            for i, item in enumerate(self.rtsp_batch_config.polling_list):
                if not getattr(item, "rtsp_url", None):
                    raise ValueError(f"rtsp_batch_config.polling_list[{i}] 缺少 rtsp_url")
                _check_url_legal(item.rtsp_url)

            # SECURITY_SINGLE 检测该流是否有音轨；SECURITY_POLLING 仅检测第一路
            first_url = self.rtsp_batch_config.polling_list[0].rtsp_url
            self._have_audio_track = have_audio_track(first_url)
            self._source_kind = SOURCE_KIND.RTSP

        # ========= 2. OFFLINE =========
        elif self.mode == MODEL.OFFLINE:
            if not url:
                raise ValueError("OFFLINE 必须使用 url 传递地址")
            if self.rtsp_batch_config:
                raise ValueError("OFFLINE 不能传入 rtsp_batch_config")

            logger.info("OFFLINE模式将忽略local_vlm_runtime_machine配置")
            _check_url_legal(url)
            self.url = ur
            self._source_kind = _determine_source_kind(url)
            self._have_audio_track = have_audio_track(url)

        else:
            raise ValueError(f"不支持的 mode: {self.mode}")

        # ---------- 云端 VLM 时，强制关闭本地 runtime machine ----------
        if self.mode in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING) and self.vlm_config.vlm_backend == "cloud":
            logger.info("VLM推理客户端选择云端，将不启用 LocalVlmRuntimeMachine")
            self.runtime_machine_config.local_vlm_runtime_machine = False

        # ---------- 初始化本地 VLM 运行时状态机（仅安防 + 本地后端 + 启用开关） ----------
        if (
            self.runtime_machine_config.local_vlm_runtime_machine
            and self.enable_b
            and self.mode in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING)
            and self.vlm_config.vlm_backend == "local"
        ):
            # 从 cut_config 中取一个基线的切片时长，兜底 10 秒
            base_cut_window_sec = float(getattr(self.cut_config, "cut_window_sec", 10.0) or 10.0)
            if base_cut_window_sec <= 0:
                base_cut_window_sec = 10.0
                logger.warning(
                    "[主控] cut_config.cut_window_sec 无效，基线切片时长回退为 %.1fs",
                    base_cut_window_sec,
                )

            # SECURITY_POLLING 模式才需要动态调轮询间隔
            if self.mode == MODEL.SECURITY_POLLING:
                update_interval_cb: Optional[Callable[[float], None]] = self.update_polling_batch_interval
            else:
                update_interval_cb = None

            try:
                self._vlm_runtime_machine = LocalVlmRuntimeMachine(
                    mode=self.mode,
                    base_cut_window_sec=base_cut_window_sec,
                    rtsp_batch_config=self.rtsp_batch_config,
                    update_cut_window_sec=self.update_cut_window_sec,
                    update_polling_batch_interval=update_interval_cb,
                )
                logger.info(
                    "[主控] LocalVlmRuntimeMachine 已启用：mode=%s, base_cut_window_sec=%.1fs",
                    self.mode.value,
                    base_cut_window_sec,
                )
            except Exception as e:
                # 为安全起见：运行时状态机初始化失败不影响主流程，只是降级不用自适应
                self._vlm_runtime_machine = None
                logger.warning("[主控] LocalVlmRuntimeMachine 初始化失败，自动降级关闭：%s", e)
        else:
            logger.info(
                "[主控] LocalVlmRuntimeMachine 未启用：mode=%s, enable_b=%s, vlm_backend=%s, switch=%s",
                self.mode.value,
                self.enable_b,
                self.vlm_config.vlm_backend,
                self.runtime_machine_config.local_vlm_runtime_machine,
            )

        # ---------- 回调占位 ----------
        self.on_vlm_delta: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_vlm_done: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_asr_no_speech: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_asr_delta: Optional[Callable[[Dict[str, Any]], None]] = None
        self.on_asr_done: Optional[Callable[[Dict[str, Any]], None]] = None

        if self.runtime_machine_config.queue_runtime_machine:
            self._queue_runtime_machine = QueueRuntimeMachine()  # 先占位, 还没实现

        # ---------- 运行期状态 ----------
        self._run_lock = threading.Lock()
        self._running = False
        self._is_paused = False  # 主控级“是否处于暂停态”的软标记

        self._init_runtime_state()

        if self.mode == MODEL.OFFLINE and self._source_kind == SOURCE_KIND.AUDIO_FILE:
            logger.info(
                "初始化离线音频分析: mode=%s, url=%s, asr_backend=%s, queue_runtime_machine=%s",
                self.mode.value, self.url, self.asr_config.asr_backend, self.runtime_machine_config.queue_runtime_machine,
            )
        elif self.mode == MODEL.OFFLINE and self._source_kind == SOURCE_KIND.VIDEO_FILE:
            logger.info(
                "初始化离线视频分析： mode=%s, url=%s, enable_b=%s, vlm_backend=%s, has_audio=%s, enable_c=%s, "
                "asr_backend=%s, queue_runtime_machine=%s",
                self.mode.value, self.url, self.enable_b, self.vlm_config.vlm_backend, self._have_audio_track,
                self.enable_c, self.asr_config.asr_backend, self.runtime_machine_config.queue_runtime_machine,
            )
        elif self.mode == MODEL.SECURITY_SINGLE and self._source_kind == SOURCE_KIND.RTSP:
            logger.info(
                "初始化单流常驻安防: mode=%s, url=%s, enable_b=%s, vlm_backend=%s, has_audio=%s, enable_c=%s, asr_backend=%s, "
                "queue_runtime_machine=%s, local_vlm_runtime_machine=%s",
                self.mode.value, self.rtsp_batch_config.polling_list[0].rtsp_url, self.enable_b, self.vlm_config.vlm_backend,
                self._have_audio_track, self.enable_c, self.asr_config.asr_backend, self.runtime_machine_config.queue_runtime_machine,
                self.runtime_machine_config.local_vlm_runtime_machine,
            )
        elif self.mode == MODEL.SECURITY_POLLING and self._source_kind == SOURCE_KIND.RTSP:
            logger.info(
                "初始化多流轮询安防: mode=%s, polling_size=%d, enable_b=%s, vlm_backend=%s, polling_batch_interval=%d, "
                "local_vlm_runtime_machine=%s",
                self.mode.value, len(self.rtsp_batch_config.polling_list), self.enable_b, self.vlm_config.vlm_backend,
                self.rtsp_batch_config.polling_batch_interval, self.runtime_machine_config.local_vlm_runtime_machine,
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
                "segments_stream": 0,
                "segments_nonstream": 0,
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

            if self._source_kind == SOURCE_KIND.AUDIO_FILE and self.mode == MODEL.OFFLINE:
                logger.info("[主控] 离线音频流程启动")
                self._start_audio_file_streaming_analyze()
            elif self._source_kind == SOURCE_KIND.VIDEO_FILE and self.mode == MODEL.OFFLINE:
                logger.info("[主控] 离线视频流程启动")
                self._start_video_file_streaming_analyze()
            elif self.mode == MODEL.SECURITY_SINGLE:
                logger.info("[主控] 单流常驻安防启动")
                self._start_single_rtsp_analyze()
            elif self.mode == MODEL.SECURITY_POLLING:
                logger.info("[主控] 多流轮询安防启动")
                self._start_polling_rtsp_analyze()
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
        """
        纯音频 OFFLINE：只启动 A + C；不启动 B，也不注入 VIDEO STOP 哨兵。
        """
        t_a, t_b, t_c = self._spawn_threads()

        to_start = [t_a]
        if self.enable_c and t_c:
            to_start.append(t_c)

        self._start_threads(*to_start)

        self._broadcast_ctrl({"type": "START"})
        self._broadcast_ctrl({"type": "MODE_CHANGE", "value": self.mode.value})

        self._watch_branch(
            t_a,
            t_b=None,
            t_c=(t_c if self.enable_c else None),
            need_video_sentinel=False,
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

        self._watch_branch(
            t_a,
            t_b=(t_b if self.enable_b else None),
            t_c=(t_c if (self._have_audio_track and self.enable_c) else None),
            need_video_sentinel=self.enable_b,
            need_audio_sentinel=(self._have_audio_track and self.enable_c),
        )

    def _start_single_rtsp_analyze(self):
        if not self.rtsp_batch_config or not self.rtsp_batch_config.polling_list:
            raise ValueError("SECURITY_SINGLE 需要有效的 rtsp_batch_config.polling_list")

        t_a, t_b, t_c = self._spawn_threads()

        to_start = [t_a]
        if self.enable_b and t_b:
            to_start.append(t_b)
        if self.enable_c and self._have_audio_track and t_c:
            to_start.append(t_c)

        self._start_threads(*to_start)

        self._broadcast_ctrl({"type": "START"})
        self._broadcast_ctrl({"type": "MODE_CHANGE", "value": self.mode.value})

        self._watch_branch(
            t_a,
            t_b=(t_b if self.enable_b else None),
            t_c=(t_c if (self.enable_c and self._have_audio_track) else None),
            need_video_sentinel=self.enable_b,
            need_audio_sentinel=(self.enable_c and self._have_audio_track),
        )

    def _start_polling_rtsp_analyze(self):
        if not self.rtsp_batch_config or not self.rtsp_batch_config.polling_list or \
                len(self.rtsp_batch_config.polling_list) < 2:
            raise ValueError("SECURITY_POLLING 需要至少 2 条流的 rtsp_batch_config.polling_list")

        t_a, t_b, t_c = self._spawn_threads()

        to_start = [t_a]
        if self.enable_b and t_b:
            to_start.append(t_b)
        if self.enable_c and self._have_audio_track and t_c:
            to_start.append(t_c)

        self._start_threads(*to_start)

        self._broadcast_ctrl({"type": "START"})
        self._broadcast_ctrl({"type": "MODE_CHANGE", "value": self.mode.value})

        self._watch_branch(
            t_a,
            t_b=(t_b if self.enable_b else None),
            t_c=(t_c if (self.enable_c and self._have_audio_track) else None),
            need_video_sentinel=self.enable_b,
            need_audio_sentinel=(self.enable_c and self._have_audio_track),
        )

    # ---------- 线程管理 ----------
    def _spawn_threads(self,):
        have_audio_for_a = bool(self._have_audio_track and self.enable_c)

        is_offline_audio = (self.mode == MODEL.OFFLINE and self._source_kind == SOURCE_KIND.AUDIO_FILE)
        # 纯音频 OFFLINE：不给 A 的视频队列
        q_video_for_a = None if is_offline_audio else (self._Q_VIDEO if self.enable_b else None)

        t_a = threading.Thread(
            target=worker_a_cut.worker_a_cut, daemon=True,
            args=(
                getattr(self, "url", None),
                self.rtsp_batch_config,
                have_audio_for_a,
                self.mode,
                (self._Q_AUDIO if self.enable_c else None),
                q_video_for_a,
                self._Q_CTRL_A,
                self._STOP,
                self.cut_config,
            ),
            name="A-切片标准化"
        )

        t_b = None
        # 纯音频 OFFLINE：不创建 B
        if self.enable_b and (not is_offline_audio):
            t_b = threading.Thread(
                target=worker_b_vlm.worker_b_vlm, daemon=True,
                args=(self._Q_VIDEO, self._Q_VLM, self._Q_CTRL_B, self._STOP, self.mode, self.rtsp_batch_config, self.vlm_config),
                name="B-VLM解析"
            )

        t_c = None
        if self.enable_c:
            t_c = threading.Thread(
                target=worker_c_asr.worker_c_asr, daemon=True,
                args=(self._Q_AUDIO, self._Q_ASR, self._Q_CTRL_C, self._STOP, self.asr_config),
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

    # ---------- 主控向 A 发 RTSP 动态控制 ----------
    def _send_rtsp_mode_message_to_a(self, msg: Dict[str, Any]):
        try:
            self._Q_CTRL_A.put_nowait(msg)
        except queue.Full:
            self._Q_CTRL_A.put(msg)

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

            if item.get("type") == "vlm_stream_delta":
                self._emit_vlm_delta(item)
                with self._stats_lock:
                    self._stats["vlm"]["deltas"] += 1
                self._emit_to_event_bus(item, channel="vlm")

            elif item.get("type") == "vlm_stream_done":
                self._emit_vlm_done(item)
                text_len = len(item.get("full_text") or "")
                try:
                    lat = float(item.get("latency_ms") or 0.0)
                except Exception:
                    lat = 0.0
                streaming_flag = bool(item.get("streaming"))
                with self._stats_lock:
                    self._stats["vlm"]["segments"] += 1
                    if streaming_flag:
                        self._stats["vlm"]["segments_stream"] += 1
                    else:
                        self._stats["vlm"]["segments_nonstream"] += 1
                    self._stats["vlm"]["text_chars"] += text_len
                    self._stats["vlm"]["latency_ms_sum"] += lat
                    if lat > self._stats["vlm"]["latency_ms_max"]:
                        self._stats["vlm"]["latency_ms_max"] = lat

                # 推到对外事件总线
                self._emit_to_event_bus(item, channel="vlm")

                # 通知本地 VLM 运行时状态机做自适应调参
                if self._vlm_runtime_machine is not None:
                    try:
                        self._vlm_runtime_machine.on_vlm_done(item)
                    except Exception as e:
                        logger.warning(
                            "[主控] 调用 LocalVlmRuntimeMachine.on_vlm_done 异常：%s",
                            e,
                        )
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

            if item.get("type") == "asr_stream_delta":
                self._emit_asr_delta(item)
                with self._stats_lock:
                    self._stats["asr"]["text_chars"] += len(item.get("delta") or "")

            elif item.get("type") == "asr_stream_done":
                self._emit_asr_done(item)
                with self._stats_lock:
                    self._stats["asr"]["segments"] += 1
                    self._stats["asr"]["text_chars"] += len(item.get("full_text") or "")
                self._emit_to_event_bus(item, channel="asr")

            elif item.get("type") == "asr_stream_no_speech":
                self._emit_asr_no_speech(item)

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
        if callable(self.on_vlm_done):
            try:
                self.on_vlm_done(payload)
            except Exception as e:
                logger.warning("[OUT] on_vlm_done 回调异常：%s", e)

        text = (payload.get("full_text") or "").strip()
        suppressed = bool(payload.get("suppressed_dup"))
        streaming_flag = payload.get("streaming")
        suppress_empty_log = os.getenv("VLM_LOG_SUPPRESS_EMPTY", "1") == "1"

        if suppressed and not text:
            if suppress_empty_log:
                return
            else:
                logger.info(
                    "[✨✨✨VLM无新增 seg#%s kind=%s ms=%s streaming=%s] (与历史一致，已省略)",
                    payload.get("segment_index"),
                    payload.get("media_kind"),
                    payload.get("latency_ms"),
                    streaming_flag,
                )
                return

        logger.info(
            "[✨✨✨VLM完整文本 seg#%s kind=%s ms=%s streaming=%s]%s%s",
            payload.get("segment_index"),
            payload.get("media_kind"),
            payload.get("latency_ms"),
            streaming_flag,
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
                    "[主控] VLM统计：segments=%d (stream=%d, nonstream=%d), deltas=%d, text_chars=%d, latency_avg=%.1fms, latency_max=%.1fms",
                    vlm["segments"], vlm["segments_stream"], vlm["segments_nonstream"],
                    vlm["deltas"], vlm["text_chars"], avg_lat, vlm["latency_ms_max"]
                )
            if "asr" in stats:
                asr = stats["asr"]
                logger.info(
                    "[主控] ASR统计：segments=%d, text_chars=%d, no_speech_segments=%d, no_speech_duration=%.2fs",
                    asr["segments"], asr["text_chars"], asr["no_speech_segments"], asr["no_speech_duration_s"]
                )
        except Exception:
            pass

        self._log_lingering_threads(where="优雅停止")
        self._events_done.set()
        logger.info("[主控] 全部线程结束或已交由进程回收")

    # ---------- 监控 ----------
    def _start_monitor(self, interval: float = 15.0):
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
                self._stats["vlm"]["segments_stream"] = 0
                self._stats["vlm"]["segments_nonstream"] = 0
                self._stats["vlm"]["deltas"] = 0
                self._stats["vlm"]["text_chars"] = 0
                self._stats["vlm"]["latency_ms_sum"] = 0.0
                self._stats["vlm"]["latency_ms_max"] = 0.0

    def _log_lingering_threads(self, where: str = "收尾阶段") -> None:
        try:
            threads = list(self._threads or []) + list(self._consumer_threads or [])
            alive = [t.name for t in threads if t and t.is_alive()]
            if alive:
                logger.warning("[主控] %s仍存活线程：%s", where, ", ".join(alive))
            else:
                logger.info("[主控] %s没有残留线程。", where)
        except Exception as e:
            logger.debug("[主控] 残留线程检查出错（忽略）：%s", e)

    # ======================== 原始流式输出生成器 ========================
    def run(self, *, max_secs: float | None = None) -> Iterator[Dict[str, Any]]:
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

    # ---------- 强停 ----------
    def _drain_queue_completely(self, q:queue.Queue, max_batch: int = 1000) -> int:
        dropped = 0
        if q is None:
            return 0
        for _ in range(max_batch):
            try:
                q.get_nowait()
                dropped += 1
            except Exception:
                break
        return dropped

    def _drain_then_inject_stop(self, q:queue.Queue, stop_obj:object):
        if q is None:
            return
        self._drain_queue_completely(q, max_batch=100000)
        for _ in range(5):
            try:
                q.put_nowait(stop_obj)
                return
            except Exception:
                try:
                    q.get_nowait()
                except Exception:
                    time.sleep(0.01)
        try:
            q.put(stop_obj, timeout=0.2)
        except Exception:
            pass

    def force_stop(self, reason: Optional[str] = "无"):
        if getattr(self, "_stopped", False):
            logger.info("[主控] force_stop() 已调用过，本次忽略。")
            return
        self._stopped = True

        logger.info(f"[主控] 外部强停触发，原因：{reason}")

        try:
            self._broadcast_ctrl({"type": "STOP", "reason": reason})
        except Exception as e:
            logger.warning(f"[主控] 广播 STOP 失败（忽略）：{e}")

        try:
            if self._Q_VIDEO is not None:
                self._drain_then_inject_stop(self._Q_VIDEO, self._STOP)
            if self._Q_AUDIO is not None:
                self._drain_then_inject_stop(self._Q_AUDIO, self._STOP)
        except Exception:
            pass

        try:
            if self.enable_b and (self._Q_VLM is not None):
                self._drain_then_inject_stop(self._Q_VLM, self._STOP)
        except Exception:
            pass
        try:
            if self.enable_c and (self._Q_ASR is not None):
                self._drain_then_inject_stop(self._Q_ASR, self._STOP)
        except Exception:
            pass

        try:
            self._drain_queue_completely(self._Q_EVENTS, max_batch=200000)
        except Exception:
            pass

        try:
            self._stop_monitor()
        except Exception:
            pass

        for t in list(self._threads or []):
            try:
                if t and t.is_alive():
                    t.join(timeout=2.0)
            except Exception:
                pass

        try:
            if self.enable_b and (self._Q_VLM is not None):
                self._drain_then_inject_stop(self._Q_VLM, self._STOP)
            if self.enable_c and (self._Q_ASR is not None):
                self._drain_then_inject_stop(self._Q_ASR, self._STOP)
        except Exception:
            pass

        for t in list(self._consumer_threads or []):
            try:
                if t and t.is_alive():
                    t.join(timeout=1.0)
            except Exception:
                pass

        try:
            self._events_done.set()
        except Exception:
            pass

        all_threads = list(self._threads or []) + list(self._consumer_threads or [])
        wait_deadline = time.time() + 3.0
        while time.time() < wait_deadline:
            alive = [t.name for t in all_threads if t and t.is_alive()]
            if not alive:
                break
            time.sleep(0.05)

        alive = [t.name for t in all_threads if t and t.is_alive()]
        if not alive:
            logger.info("[主控] 强停完成：所有线程已退出，队列已清空并注入 STOP 哨兵。")
        else:
            logger.info("[主控] 强停完成：仍有存活线程（超时未等齐）：%s", alive)

    # ====================== 对外控制接口 ======================

    def pause(self):
        """暂停 A/B/C。"""
        if not self._running:
            logger.warning("[主控] pause() 调用但当前未运行，已忽略。")
            return
        self._is_paused = True
        self._broadcast_ctrl({"type": "PAUSE"})
        logger.info("[主控] 已广播 PAUSE")

    def resume(self):
        """
        恢复 A/B/C。
        - OFFLINE：不清空队列（A 会继续旧 t0）
        - SECURITY_SINGLE / SECURITY_POLLING：先清空 A→(B/C) 的 VIDEO/AUDIO 队列，确保恢复后只消费最新画面/音频
        """
        if not self._running:
            logger.warning("[主控] resume() 调用但当前未运行，已忽略。")
            return
        if self.mode in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING):
            # 双保险：主控侧先清一次（A 侧恢复时也会清）
            dropped_v = self._drain_queue_completely(self._Q_VIDEO, max_batch=200000)
            dropped_a = self._drain_queue_completely(self._Q_AUDIO, max_batch=200000)
            if dropped_v or dropped_a:
                logger.info("[主控] RESUME 前已清空下游队列：VIDEO=%d, AUDIO=%d", dropped_v, dropped_a)
        self._is_paused = False
        self._broadcast_ctrl({"type": "RESUME"})
        logger.info("[主控] 已广播 RESUME")

    def polling_add_stream(self, add_rtsp: RTSP):
        """
        SECURITY_POLLING 下动态新增一路 RTSP（上限 50）。
        - 会更新本地 rtsp_batch_config.polling_list
        - 向 A 发送 {type: 'RTSP_ADD_STREAM', item: {...}} 控制消息
        """
        if self.mode != MODEL.SECURITY_POLLING:
            raise RuntimeError("仅在 SECURITY_POLLING 模式下支持动态新增流")
        if not self.rtsp_batch_config:
            raise RuntimeError("rtsp_batch_config 未初始化")

        # --- 基本字段取值与校验 ---
        rtsp_url = getattr(add_rtsp, "rtsp_url", None)
        if not rtsp_url or not isinstance(rtsp_url, str):
            raise ValueError("add_rtsp.rtsp_url 必须为非空字符串")
        _check_url_legal(rtsp_url)

        # 规范化切片并发数（至少 1）
        if hasattr(add_rtsp, "rtsp_cut_number"):
            try:
                cut_num = int(getattr(add_rtsp, "rtsp_cut_number", 1))
            except Exception:
                cut_num = 1
            cut_num = max(1, cut_num)
            try:
                setattr(add_rtsp, "rtsp_cut_number", cut_num)
            except Exception:
                # 若是冻结模型也不强求就地改；至少下面发消息时会使用规范化后的数值
                pass
        else:
            # 若没有该字段，消息里兜底为 1
            cut_num = 1

        # --- 去重与上限 ---
        urls = [getattr(x, "rtsp_url", None) for x in self.rtsp_batch_config.polling_list]
        if rtsp_url in urls:
            raise ValueError(f"已存在重复的 rtsp_url：{rtsp_url}")

        if len(self.rtsp_batch_config.polling_list) >= 50:
            raise ValueError("SECURITY_POLLING 下最多 50 路，已达上限")

        # --- 本地追加（保持 RTSP 类型）---
        self.rtsp_batch_config.polling_list.append(add_rtsp)
        logger.info("[主控] 本地已新增流：%s（当前共 %d 路）",
                    rtsp_url, len(self.rtsp_batch_config.polling_list))

        # --- 序列化成可下发的 dict ---
        if hasattr(add_rtsp, "model_dump") and callable(add_rtsp.model_dump):
            item_dict = add_rtsp.model_dump()
        elif hasattr(add_rtsp, "dict") and callable(add_rtsp.dict):
            item_dict = add_rtsp.dict()
        else:
            item_dict = {
                "rtsp_url": rtsp_url,
                "rtsp_prompt": getattr(add_rtsp, "rtsp_prompt", None),
                "rtsp_cut_number": cut_num,
            }
            # 兼容可能存在的其它字段（可按需补充）
            for extra_key in ("name", "camera_id", "tags", "extra"):
                if hasattr(add_rtsp, extra_key):
                    item_dict[extra_key] = getattr(add_rtsp, extra_key)

        # --- 通知 A 线程实时增补 ---
        self._send_rtsp_mode_message_to_a({
            "type": "RTSP_ADD_STREAM",
            "item": item_dict
        })

    def polling_remove_stream(self, *, index: Optional[int] = None, rtsp_url: Optional[str] = None):
        """
        SECURITY_POLLING 下动态删除一路 RTSP（下限 2）。
        - 支持按 index 或 rtsp_url 定位（必须二选一）
        - 会更新本地 rtsp_batch_config.polling_list
        - 向 A 发送 {type: 'RTSP_REMOVE_STREAM', match: {'index': i}} 或 {'url': url}
        """
        if self.mode != MODEL.SECURITY_POLLING:
            raise RuntimeError("仅在 SECURITY_POLLING 模式下支持动态删除流")
        if not self.rtsp_batch_config:
            raise RuntimeError("rtsp_batch_config 未初始化")
        if (index is None) == (rtsp_url is None):
            raise ValueError("polling_remove_stream 需要在 index 与 rtsp_url 之间二选一")

        n = len(self.rtsp_batch_config.polling_list)
        if n <= 2:
            raise ValueError("SECURITY_POLLING 下最少保留 2 路，无法继续删除")

        # 解析要删的 index
        del_idx = None
        if index is not None:
            if not (0 <= int(index) < n):
                raise IndexError(f"index 越界: 0~{(n-1)}")
            del_idx = int(index)
            url = getattr(self.rtsp_batch_config.polling_list[del_idx], "rtsp_url", None)
        else:
            # by url
            for i, it in enumerate(self.rtsp_batch_config.polling_list):
                if getattr(it, "rtsp_url", None) == rtsp_url:
                    del_idx = i
                    break
            if del_idx is None:
                raise ValueError(f"未找到要删除的 rtsp_url：{rtsp_url}")
            url = rtsp_url

        # 本地删除
        removed = self.rtsp_batch_config.polling_list.pop(del_idx)
        logger.info("[主控] 本地已删除流：%s（当前共 %d 路）", getattr(removed, "rtsp_url", None), len(self.rtsp_batch_config.polling_list))

        # 通知 A 线程实时移除
        match = ({"index": del_idx} if index is not None else {"url": url})
        self._send_rtsp_mode_message_to_a({
            "type": "RTSP_REMOVE_STREAM",
            "match": match
        })

    def update_stream_rtsp(self, current_rtsp_url: str, new_rtsp: RTSP) -> int:
        """
        SECURITY_SINGLE / SECURITY_POLLING 下：
        - 将 polling_list 中“匹配 current_rtsp_url 的条目”用 new_rtsp 原地替换（保持索引不变）
        - 本地更新后，给 A 线程发送 {type: 'RTSP_UPDATE_STREAM', old_rtsp_url, item: {...}}
        返回：被替换的索引
        """
        if self.mode not in (MODEL.SECURITY_SINGLE, MODEL.SECURITY_POLLING):
            raise RuntimeError("仅在 SECURITY_SINGLE / SECURITY_POLLING 模式下支持动态修改流配置")
        if not self.rtsp_batch_config:
            raise RuntimeError("rtsp_batch_config 未初始化")

        # 校验存在
        plist = self.rtsp_batch_config.polling_list
        idx = None
        for i, it in enumerate(plist):
            if getattr(it, "rtsp_url", None) == current_rtsp_url:
                idx = i
                break
        if idx is None:
            raise ValueError(f"未找到待更新的流：{current_rtsp_url}")

        # 校验新对象
        new_url = getattr(new_rtsp, "rtsp_url", None)
        if not new_url or not isinstance(new_url, str):
            raise ValueError("new_rtsp.rtsp_url 必须为非空字符串")
        _check_url_legal(new_url)

        # 去重：除了自己所在位置，其他位置不能重复
        for j, it in enumerate(plist):
            if j == idx:
                continue
            if getattr(it, "rtsp_url", None) == new_url:
                raise ValueError(f"新的 rtsp_url 已存在于列表的其他位置：{new_url}")

        # 规范化 cut_number（至少 1）
        try:
            cn = int(getattr(new_rtsp, "rtsp_cut_number", 1))
        except Exception:
            cn = 1
        if cn < 1:
            try:
                setattr(new_rtsp, "rtsp_cut_number", 1)
            except Exception:
                pass

        # 本地原地替换
        self.rtsp_batch_config.polling_list[idx] = new_rtsp
        logger.info("[主控] 已更新流：%s -> %s（保持索引 %d 不变）",
                    current_rtsp_url, new_url, idx)

        # 序列化成可下发的 dict
        if hasattr(new_rtsp, "model_dump") and callable(new_rtsp.model_dump):
            item_dict = new_rtsp.model_dump()
        elif hasattr(new_rtsp, "dict") and callable(new_rtsp.dict):
            item_dict = new_rtsp.dict()
        else:
            item_dict = {
                "rtsp_url": new_url,
                "rtsp_system_prompt": getattr(new_rtsp, "rtsp_system_prompt", None),
                "rtsp_cut_number": int(getattr(new_rtsp, "rtsp_cut_number", 1) or 1),
            }

        # 通知 A 线程
        self._send_rtsp_mode_message_to_a({
            "type": "RTSP_UPDATE_STREAM",
            "old_rtsp_url": current_rtsp_url,
            "item": item_dict,
            "index": idx,       # 便于 A 侧选择“按 index”更新（实现里两种方式都可）
        })
        return idx

    def update_polling_batch_interval(self, new_interval: float) -> None:
        """
        SECURITY_POLLING 下：运行时修改轮询间隔（秒）。
        - 本地更新 rtsp_batch_config.polling_batch_interval
        - 通知 A 线程 {type: 'RTSP_UPDATE_INTERVAL', polling_batch_interval: float}
        """
        if self.mode != MODEL.SECURITY_POLLING:
            raise RuntimeError("仅在 SECURITY_POLLING 模式下支持修改 polling_batch_interval")
        if not self.rtsp_batch_config:
            raise RuntimeError("rtsp_batch_config 未初始化")
        if new_interval < 10.0:
            raise ValueError("polling_batch_interval 不得小于 10 秒")

        self.rtsp_batch_config.polling_batch_interval = new_interval
        logger.info("[主控] 轮询间隔已更新为 %d s", new_interval)

        self._send_rtsp_mode_message_to_a({
            "type": "RTSP_UPDATE_INTERVAL",
            "polling_batch_interval": new_interval
        })

    def update_cut_window_sec(self,new_cut_window_sec:float) -> None:
        if self.mode not in (MODEL.SECURITY_SINGLE,MODEL.SECURITY_POLLING):
             raise RuntimeError("仅在 SECURITY_SINGLE 和 SECURITY_POLLING 模式下支持修改 cut_window_sec")
        if not self.rtsp_batch_config:
            raise RuntimeError("cut_window_sec 未初始化")
        if new_cut_window_sec < 1.0:
            raise ValueError("cut_window_sec 不得小于 1 秒")

        self._send_rtsp_mode_message_to_a({
            "type": "RTSP_UPDATE_CUT_WINDOW_SEC",
            "cut_window_sec": new_cut_window_sec
        })


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
