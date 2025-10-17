# -*- coding: utf-8 -*-
from __future__ import annotations

import os
import time
import queue
import wave
import logging
import threading
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, List, Tuple

logger = logging.getLogger("src.workers.worker_c_asr")

# ----------- 可选：WebRTC VAD -----------
try:
    import webrtcvad  # type: ignore
    _HAVE_WEBRTCVAD = True
except Exception:
    _HAVE_WEBRTCVAD = False

# ----------- Paraformer SDK -----------
try:
    import dashscope  # type: ignore
    from dashscope.audio.asr import Recognition, RecognitionCallback  # type: ignore
    _HAVE_PARA = True
except Exception as e:
    _HAVE_PARA = False
    _PARA_IMPORT_ERR = e


# ================= 工具 =================
def _iter_pcm_frames_from_wav(wav_path: str, *, block_bytes: int = 3200) -> Iterable[bytes]:
    """读取 16kHz/mono/PCM16 WAV，按 100ms(3200B) 切块产出纯 PCM16。"""
    with wave.open(wav_path, "rb") as wf:
        nch = wf.getnchannels()
        sbytes = wf.getsampwidth()
        rate = wf.getframerate()
        if not (nch == 1 and sbytes == 2 and rate == 16000):
            raise ValueError(f"WAV 参数不符，期待 16k/mono/PCM16，实际: ch={nch}, sw={sbytes}, sr={rate}")
        while True:
            data = wf.readframes(block_bytes // sbytes)
            if not data:
                break
            yield data


def _iso_local(ts: float) -> str:
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%dT%H:%M:%S")


# ======== 可选 VAD：优先 WebRTC，失败用能量阈值 ========
def _analyze_vad_active_ratio(
    wav_path: str,
    *,
    frame_ms: int = 20,
    aggr: int = 1,
    energy_dbfs_thresh: float = -45.0,
    min_speech_ms: int = 300,
    hangover_ms: int = 300,
) -> Dict[str, Any]:
    """
    返回：
      {
        "is_speech": bool,
        "active_ratio": float(0~1),
        "backend_used": "webrtcvad"|"energy"|"disabled",
        "applied_params": {...}
      }
    """
    applied = {
        "aggr": aggr,
        "energy_dbfs_thresh": energy_dbfs_thresh,
        "min_active_ratio": 0.08,
        "min_speech_ms": min_speech_ms,
        "hangover_ms": hangover_ms,
    }

    try:
        with wave.open(wav_path, "rb") as wf:
            nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            if not (nch == 1 and sw == 2 and sr == 16000):
                return {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": applied}
            pcm = wf.readframes(wf.getnframes())
    except Exception as e:
        logger.warning(f"[C] VAD 读取 WAV 失败，降级：{e}")
        return {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": applied}

    if _HAVE_WEBRTCVAD:
        try:
            vad = webrtcvad.Vad(int(aggr))
            frame_bytes = int(sr * frame_ms / 1000) * 2
            total = 0
            act = 0
            for i in range(0, len(pcm), frame_bytes):
                chunk = pcm[i:i + frame_bytes]
                if len(chunk) < frame_bytes:
                    break
                if vad.is_speech(chunk, sr):
                    act += 1
                total += 1
            ratio = (act / total) if total else 0.0
            return {
                "is_speech": bool(ratio >= applied["min_active_ratio"]),
                "active_ratio": float(ratio),
                "backend_used": "webrtcvad",
                "applied_params": applied
            }
        except Exception as e:
            logger.warning(f"[C] WebRTC VAD 失败，能量兜底：{e}")

    # 简易能量阈值
    try:
        import array, math
        pcm_i16 = array.array("h", pcm)
        if not pcm_i16:
            return {"is_speech": False, "active_ratio": 0.0, "backend_used": "energy", "applied_params": applied}
        frame_samples = int(sr * frame_ms / 1000) or 320
        total = 0
        act = 0
        for i in range(0, len(pcm_i16), frame_samples):
            frm = pcm_i16[i:i + frame_samples]
            if not frm:
                break
            rms = math.sqrt(sum(int(x) * int(x) for x in frm) / len(frm))
            if rms <= 1e-6:
                dbfs = -90.0
            else:
                dbfs = 20.0 * math.log10(rms / 32768.0 * 2.0)
            if dbfs > energy_dbfs_thresh:
                act += 1
            total += 1
        ratio = (act / total) if total else 0.0
        return {
            "is_speech": bool(ratio >= 0.08),
            "active_ratio": float(ratio),
            "backend_used": "energy",
            "applied_params": applied
        }
    except Exception:
        return {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": applied}


# ======== Paraformer 回调：收 sentence_end 句级结果 ========
class _ParaCallback(RecognitionCallback):
    def __init__(self):
        self.sentences: List[dict] = []  # 仅存 sentence_end=True 的句对象
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._err: Optional[str] = None

    def on_complete(self) -> None:
        self._done.set()

    def on_error(self, message) -> None:
        try:
            req_id = getattr(message, "request_id", None)
            msg = getattr(message, "message", None)
            logger.error(f"[C] Paraformer 错误: request_id={req_id}, err={msg}")
        except Exception:
            logger.error("[C] Paraformer 错误（无法解析 message）")
        self._err = "error"
        self._done.set()

    def on_event(self, result) -> None:
        try:
            sentence = result.get_sentence()  # dict
        except Exception:
            return
        if sentence and sentence.get("sentence_end", False):
            with self._lock:
                self.sentences.append(sentence)

    def wait_done(self, timeout: Optional[float]) -> bool:
        return self._done.wait(timeout=timeout)

    def fetch_sentences(self) -> List[dict]:
        with self._lock:
            out = list(self.sentences)
            self.sentences.clear()
            return out

    def has_error(self) -> bool:
        return self._err is not None


# ======== 从句对象中尽力解析开始/结束时间（ms） ========
def _extract_sentence_times_ms(s: dict) -> Tuple[Optional[float], Optional[float]]:
    """
    兼容多种潜在字段：end_time/end_ms/end、start_time/begin_time/start_ms/start、ts 等
    返回 (start_ms, end_ms)，若无则为 (None, None)
    """
    def _get_any(d, keys):
        for k in keys:
            v = d.get(k)
            if v is not None:
                try:
                    return float(v)
                except Exception:
                    pass
        return None

    end_ms = _get_any(s, ("end_time", "end_ms", "end", "etime", "ts_end"))
    if end_ms is None:
        # 有些只给一个 ts，当作 end
        end_ms = _get_any(s, ("ts", "time_ms", "time"))
    start_ms = _get_any(s, ("start_time", "begin_time", "start_ms", "start", "stime", "ts_begin"))
    return start_ms, end_ms


# ========== 主体：C 线程 ==========
def worker_c_asr(
    q_audio: "queue.Queue[Dict[str, Any]]",
    q_asr: "queue.Queue[Dict[str, Any]]",
    q_ctrl: "queue.Queue[Dict[str, Any]]",
    stop: object,
):
    """
    - 从 q_audio 取离线 wav 段，先做 VAD 预判，再推往 Paraformer 做识别。
    - 只在 sentence_end=True 时投递句级增量，段尾投收尾。
    - 每个句级增量带上 media_ts（对齐到视频时间线）与 clip_t0/clip_t1。
    """
    logger.info("[C] 线程启动: C-ASR转录")
    if _HAVE_PARA and "DASHSCOPE_API_KEY" in os.environ:
        dashscope.api_key = os.environ["DASHSCOPE_API_KEY"]

    # 配置
    asr_model = os.getenv("ASR_MODEL", "paraformer-realtime-v2")
    lang = os.getenv("ASR_LANG", "zh")
    sr_hz = 16000
    block_bytes = 3200  # 100ms

    # VAD
    vad_enabled = os.getenv("ASR_VAD_ENABLED", "1") == "1"
    vad_aggr = int(os.getenv("ASR_VAD_AGGR", "1"))
    vad_dbfs = float(os.getenv("ASR_VAD_DBFS", "-45"))
    vad_min_ms = int(os.getenv("ASR_VAD_MIN_MS", "300"))
    vad_hang_ms = int(os.getenv("ASR_VAD_HANG_MS", "300"))

    def _q_put_with_retry(obj: Dict[str, Any], *, timeout=0.2, tries=50) -> bool:
        n = 0
        while True:
            # 控制优先
            try:
                ctrl = q_ctrl.get_nowait()
                if ctrl is stop or (isinstance(ctrl, dict) and ctrl.get("type") in ("STOP", "SHUTDOWN")):
                    logger.info("[C] 收到控制队列 STOP ，退出")
                    return False
                try:
                    q_ctrl.put_nowait(ctrl)
                except queue.Full:
                    pass
            except queue.Empty:
                pass

            try:
                q_asr.put(obj, timeout=timeout)
                return True
            except queue.Full:
                n += 1
                if n >= tries:
                    logger.warning("[C] q_asr 持续拥堵，放弃投递。")
                    return False
            except Exception as e:
                logger.warning(f"[C] q_asr.put 异常，放弃：{e}")
                return False

    try:
        seg_auto_idx = 0
        while True:
            # 取音频任务或 STOP 哨兵
            try:
                item = q_audio.get(timeout=0.1)
            except queue.Empty:
                # 监听 STOP
                try:
                    msg = q_ctrl.get_nowait()
                    if msg is stop or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                        logger.info("[C] 收到控制队列 STOP ，退出")
                        return
                    try:
                        q_ctrl.put_nowait(msg)
                    except queue.Full:
                        pass
                except queue.Empty:
                    pass
                continue

            if item is stop:
                logger.info("[C] 收到数据队列 STOP，退出")
                return

            a_path = item.get("path")
            t0 = float(item.get("t0", 0.0))  # clip_t0
            t1 = float(item.get("t1", 0.0))  # clip_t1
            seg_idx = item.get("segment_index")
            if seg_idx is None:
                seg_idx = seg_auto_idx
                seg_auto_idx += 1

            if not a_path or not os.path.exists(a_path):
                logger.warning(f"[C] 音频文件不存在，跳过 seg#{seg_idx}: {a_path}")
                continue

            # ---------- VAD 预判 ----------
            vad_info = {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": {}}
            if vad_enabled:
                try:
                    vad_info = _analyze_vad_active_ratio(
                        a_path, aggr=vad_aggr, energy_dbfs_thresh=vad_dbfs,
                        min_speech_ms=vad_min_ms, hangover_ms=vad_hang_ms
                    )
                except Exception as e:
                    logger.warning(f"[C] VAD 失败，降级直接转写：{e}")

            if not vad_info.get("is_speech", True):
                now_ts = time.time()
                ok = _q_put_with_retry({
                    "type": "asr_stream_no_speech",
                    "segment_index": seg_idx,
                    "full_text": "[SKIPPED_NO_SPEECH]",
                    "usage": {"vad": vad_info, "silence_hint": item.get("silence_hint")},
                    "model": asr_model,
                    "latency_ms": 0,
                    "lang": lang,
                    "sr_hz": sr_hz,
                    "t0": t0, "t1": t1,
                    "produce_ts": now_ts,
                    "produce_iso": _iso_local(now_ts),
                })
                if not ok:
                    return
                continue

            if not _HAVE_PARA:
                now_ts = time.time()
                logger.error(f"[C] 未安装/导入 Paraformer SDK（dashscope）失败：{_PARA_IMPORT_ERR}")
                ok = _q_put_with_retry({
                    "type": "asr_stream_no_speech",
                    "segment_index": seg_idx,
                    "full_text": "[SKIPPED_SDK_NOT_AVAILABLE]",
                    "usage": {"vad": vad_info, "silence_hint": item.get("silence_hint")},
                    "model": "N/A",
                    "latency_ms": 0,
                    "lang": lang,
                    "sr_hz": sr_hz,
                    "t0": t0, "t1": t1,
                    "produce_ts": now_ts,
                    "produce_iso": _iso_local(now_ts),
                })
                if not ok:
                    return
                continue

            # ---------- Paraformer：仅句级 ----------
            cb = _ParaCallback()
            recog = Recognition(
                model=os.getenv("ASR_MODEL", "paraformer-realtime-v2"),
                format='pcm',
                sample_rate=sr_hz,
                semantic_punctuation_enabled=True,
                callback=cb
            )

            start_ts = time.time()
            try:
                recog.start()
                for pcm in _iter_pcm_frames_from_wav(a_path, block_bytes=block_bytes):
                    # 控制中断
                    try:
                        ctrl = q_ctrl.get_nowait()
                        if ctrl is stop or (isinstance(ctrl, dict) and ctrl.get("type") in ("STOP", "SHUTDOWN")):
                            logger.info("[C] 收到控制队列 STOP ，退出")
                            return
                        try:
                            q_ctrl.put_nowait(ctrl)
                        except queue.Full:
                            pass
                    except queue.Empty:
                        pass

                    recog.send_audio_frame(pcm)
                recog.stop()
            except Exception as e:
                logger.warning(f"[C] Paraformer 发送/停止异常：{e}")

            # 等待 SDK 完成，取句级结果
            cb.wait_done(timeout=float(os.getenv("ASR_PARA_WAIT_DONE_S", "10.0")))
            if cb.has_error():
                logger.warning("[C] Paraformer 识别报错（已记录），继续后续段。")

            sentences = cb.fetch_sentences()  # 仅 sentence_end=True
            n_sent = len(sentences)
            dur = max(0.0, t1 - t0) if (t1 >= t0) else 0.0

            # 先预解析每句 (start_ms, end_ms)
            sent_ms_list: List[Tuple[Optional[float], Optional[float]]] = [
                _extract_sentence_times_ms(s) for s in sentences
            ]

            # 如果完全没有时间戳，备用线性分布：把每句的 end 均分到 [t0,t1]
            linear_end_ts = []
            if n_sent > 0:
                for i in range(1, n_sent + 1):
                    # i/n_sent 百分位
                    frac = i / n_sent
                    linear_end_ts.append(t0 + frac * dur)

            # 投句级增量（带 media_ts）
            seq = 0
            concat_texts: List[str] = []
            sentence_times_abs: List[Dict[str, Optional[float]]] = []

            for idx, s in enumerate(sentences):
                text = (s.get("text") or "").strip()
                if not text:
                    continue

                start_ms, end_ms = sent_ms_list[idx]
                # 计算绝对时间
                if end_ms is not None:
                    media_ts = t0 + float(end_ms) / 1000.0
                elif start_ms is not None:
                    media_ts = t0 + float(start_ms) / 1000.0
                else:
                    media_ts = linear_end_ts[idx] if (idx < len(linear_end_ts)) else (t0 + 0.5 * dur)

                # 记录句级时间范围（若有）
                sent_abs = {
                    "start_ts": (t0 + float(start_ms) / 1000.0) if start_ms is not None else None,
                    "end_ts":   (t0 + float(end_ms) / 1000.0)   if end_ms is not None else None,
                }
                sentence_times_abs.append(sent_abs)

                concat_texts.append(text)
                seq += 1
                now_ts = time.time()

                ok = _q_put_with_retry({
                    "type": "asr_stream_delta",
                    "segment_index": seg_idx,
                    "seq": seq,
                    "delta": text,                  # 句级增量 = 本句全文
                    "usage": {"vad": vad_info, "silence_hint": item.get("silence_hint")},
                    "model": asr_model,
                    "lang": lang,
                    "sr_hz": sr_hz,
                    "t0": t0, "t1": t1,            # 片段边界
                    "media_ts": float(media_ts),    # 对应视频时间（秒，绝对）
                    "media_ts_iso": _iso_local(media_ts),
                    "produce_ts": now_ts,           # 发射时刻
                    "produce_iso": _iso_local(now_ts),
                })
                if not ok:
                    return

            # 段尾收尾
            latency_ms = int((time.time() - start_ts) * 1000)
            full_text = "".join(concat_texts)
            now_ts = time.time()
            ok = _q_put_with_retry({
                "type": "asr_stream_done",
                "segment_index": seg_idx,
                "full_text": full_text,
                "usage": {"vad": vad_info, "silence_hint": item.get("silence_hint")},
                "model": asr_model,
                "latency_ms": latency_ms,
                "lang": lang,
                "sr_hz": sr_hz,
                "t0": t0, "t1": t1,
                "sentence_times": sentence_times_abs,  # 每句的绝对时间范围（若 SDK 提供）
                "produce_ts": now_ts,
                "produce_iso": _iso_local(now_ts),
            })
            if not ok:
                return

            logger.info("[C] Paraformer 识别完成 seg#%s", seg_idx)

    except Exception as e:
        logger.exception(f"[C] 异常退出：{e}")
    finally:
        logger.info("[C] 线程退出清理完成。")
