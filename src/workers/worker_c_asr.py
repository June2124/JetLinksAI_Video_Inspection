'''
Author: 13594053100@163.com
Date: 2025-10-24 15:47:23
LastEditTime: 2025-11-25 15:06:02
'''

from __future__ import annotations

import os
import time
import queue
import wave
import logging
from datetime import datetime
from typing import Optional, Dict, Any, Iterable, List, Tuple

from src.configs.asr_config import AsrConfig
from src.utils.asr_client.cloud_asr_client import CloudASRClient
from src.utils.asr_client.local_sophon_asr_client import LocalASRClient

logger = logging.getLogger("src.workers.worker_c_asr")

# ----------- 可选：WebRTC VAD -----------
try:
    import webrtcvad  # type: ignore
    _HAVE_WEBRTCVAD = True
except Exception:
    _HAVE_WEBRTCVAD = False


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
) -> Dict[str, Any]:
    """
    返回：
      {
        "is_speech": bool,
        "active_ratio": float(0~1),
        "backend_used": "webrtcvad"|"energy"|"disabled",
        "applied_params": {...}
      }
    这里只做“是否有语音”的段级判断，用于跳过纯静音段；并不替代 ASR 的断句策略。
    """
    applied = {
        "aggr": aggr,
        "energy_dbfs_thresh": energy_dbfs_thresh,
        "min_active_ratio": 0.08,
    }

    try:
        with wave.open(wav_path, "rb") as wf:
            nch, sw, sr = wf.getnchannels(), wf.getsampwidth(), wf.getframerate()
            if not (nch == 1 and sw == 2 and sr == 16000):
                # 不是标准 WAV 就不做 VAD 了
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


# ========== 主体：C 线程 ==========
def worker_c_asr(
    q_audio: "queue.Queue[Dict[str, Any]]",
    q_asr: "queue.Queue[Dict[str, Any]]",
    q_ctrl: "queue.Queue[Dict[str, Any]]",
    stop: object,
    asr_config: AsrConfig
):
    """
    - 从 q_audio 取离线 wav 段，先做段级 VAD 预判，再推往云/本地 ASR。
    - 句级增量用 asr_stream_delta；段尾用 asr_stream_done。
    - 透传 A 侧元数据：stream_url/stream_index/stream_segment_index/poll_round。
    - 支持 START/PAUSE/RESUME/STOP 控制。
    """
    logger.info("[C] 线程启动: C-ASR转写")
    running = False
    paused = False

    # 基本配置（语言、采样率）
    lang = os.getenv("ASR_LANG", "zh")
    sr_hz = 16000
    block_bytes = 3200  # 100ms

    # 断句策略由 asr_config 决定：语义断句/逆文本/去填充词/标点预测/静音阈值
    asr_opts = dict(
        semantic_punctuation_enabled=asr_config.semantic_punctuation_enabled,
        disfluency_removal_enabled=asr_config.disfluency_removal_enabled,
        max_sentence_silence=asr_config.max_sentence_silence,
        punctuation_prediction_enabled=asr_config.punctuation_prediction_enabled,
        inverse_text_normalization_enabled=asr_config.inverse_text_normalization_enabled,
        sample_rate=sr_hz,
        block_bytes=block_bytes,
        wait_done_seconds=float(os.getenv("ASR_PARA_WAIT_DONE_S", "10.0")),
    )

    # 段级 VAD 预判（保留为可选，避免白跑）
    vad_enabled = os.getenv("ASR_VAD_ENABLED", "1") == "1"
    vad_aggr = int(os.getenv("ASR_VAD_AGGR", "1"))
    vad_dbfs = float(os.getenv("ASR_VAD_DBFS", "-45"))

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

    # 控制循环助手
    def _drain_ctrl_once() -> Optional[str]:
        nonlocal running, paused
        try:
            msg = q_ctrl.get_nowait()
        except queue.Empty:
            return None
        if msg is stop:
            return "STOP"
        if isinstance(msg, dict):
            typ = msg.get("type")
            if typ in ("START", "RESUME"):
                running, paused = True, False
                logger.info("[C] 启动/继续 ASR 解析")
            elif typ == "PAUSE":
                paused = True
                logger.info("[C] 暂停 ASR 解析")
            elif typ in ("STOP", "SHUTDOWN"):
                return "STOP"
        return None

    # 主循环
    try:
        seg_auto_idx = 0
        while True:
            # 控制优先
            res = _drain_ctrl_once()
            if res == "STOP":
                logger.info("[C] 收到 STOP，退出。")
                return
            if not running or paused:
                time.sleep(0.1)
                continue

            # 取音频任务或继续监听控制
            try:
                item = q_audio.get(timeout=0.1)
            except queue.Empty:
                continue

            try:
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

                # 透传元信息（供上游使用）
                stream_url = item.get("stream_url")
                stream_index = item.get("stream_index")
                stream_segment_index = item.get("stream_segment_index")
                poll_round = item.get("poll_round")

                if not a_path or not os.path.exists(a_path):
                    logger.warning(f"[C] 音频文件不存在，跳过 seg#{seg_idx}: {a_path}")
                    continue

                # ---------- 段级 VAD 预判 ----------
                vad_info = {"is_speech": True, "active_ratio": 1.0, "backend_used": "disabled", "applied_params": {}}
                if vad_enabled:
                    try:
                        vad_info = _analyze_vad_active_ratio(a_path, aggr=vad_aggr, energy_dbfs_thresh=vad_dbfs)
                    except Exception as e:
                        logger.warning(f"[C] VAD 失败，降级直接转写：{e}")

                if not vad_info.get("is_speech", True):
                    now_ts = time.time()
                    ok = _q_put_with_retry({
                        "type": "asr_stream_no_speech",
                        "segment_index": seg_idx,
                        "full_text": "[SKIPPED_NO_SPEECH]",
                        "usage": {"vad": vad_info},
                        "model": asr_config.asr_cloud_model_name.value if asr_config.asr_backend == "cloud" else asr_config.asr_local_model_name.value,
                        "latency_ms": 0,
                        "lang": lang,
                        "sr_hz": sr_hz,
                        "t0": t0, "t1": t1,
                        "produce_ts": now_ts,
                        "produce_iso": _iso_local(now_ts),
                        "stream_url": stream_url,
                        "stream_index": stream_index,
                        "stream_segment_index": stream_segment_index,
                        "poll_round": poll_round,
                    })
                    if not ok:
                        return
                    continue

                # ---------- 选择 ASR 后端 ----------
                backend = asr_config.asr_backend
                asr_model_name = asr_config.asr_cloud_model_name.value if backend == "cloud" else asr_config.asr_local_model_name.value

                want_streaming = True  # 句级增量
                t_start = time.time()
                try:
                    if backend == "cloud":
                        client = CloudASRClient(
                            model_name=asr_model_name,
                            audio_uri=a_path,
                            want_streaming=want_streaming,
                            q_ctrl=q_ctrl,
                            stop=stop,
                            asr_options=asr_opts,
                        )
                    else:
                        # 先打通接口，占位
                        client = LocalASRClient(
                            model_name=asr_model_name,
                            audio_uri=a_path,
                            q_ctrl=q_ctrl,
                            stop=stop,
                            asr_options=asr_opts,
                        )

                    mode, iter_pair, nonstream_pair = client.infer()

                except NotImplementedError as e:
                    # 本地 ASR 未实现
                    now_ts = time.time()
                    logger.error(f"[C] 本地 ASR 未实现：{e}")
                    _q_put_with_retry({
                        "type": "asr_stream_done",
                        "segment_index": seg_idx,
                        "full_text": "[LOCAL_ASR_NOT_IMPLEMENTED]",
                        "usage": {"vad": vad_info},
                        "model": asr_model_name,
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "lang": lang,
                        "sr_hz": sr_hz,
                        "t0": t0, "t1": t1,
                        "produce_ts": now_ts,
                        "produce_iso": _iso_local(now_ts),
                        "stream_url": stream_url,
                        "stream_index": stream_index,
                        "stream_segment_index": stream_segment_index,
                        "poll_round": poll_round,
                    })
                    continue
                except Exception as e:
                    # SDK 不可用或调用失败
                    now_ts = time.time()
                    logger.error(f"[C] ASR 后端调用失败（seg#{seg_idx}）：{e}")
                    _q_put_with_retry({
                        "type": "asr_stream_done",
                        "segment_index": seg_idx,
                        "full_text": f"[ASR_BACKEND_ERROR] {e}",
                        "usage": {"vad": vad_info},
                        "model": asr_model_name,
                        "latency_ms": int((time.time() - t_start) * 1000),
                        "lang": lang,
                        "sr_hz": sr_hz,
                        "t0": t0, "t1": t1,
                        "produce_ts": now_ts,
                        "produce_iso": _iso_local(now_ts),
                        "stream_url": stream_url,
                        "stream_index": stream_index,
                        "stream_segment_index": stream_segment_index,
                        "poll_round": poll_round,
                    })
                    continue

                # ---------- 发句级增量 / 段尾 ----------
                usage = None
                full_text = ""
                if mode == "stream":
                    seq = 1
                    buf: List[str] = []
                    for delta, usage_part in iter_pair:  # type: ignore
                        if delta:
                            buf.append(delta)
                            now_ts = time.time()
                            ok = _q_put_with_retry({
                                "type": "asr_stream_delta",
                                "segment_index": seg_idx,
                                "seq": seq,
                                "delta": delta,                 # 本句全文
                                "usage": {"vad": vad_info},
                                "model": asr_model_name,
                                "lang": lang,
                                "sr_hz": sr_hz,
                                "t0": t0, "t1": t1,
                                "produce_ts": now_ts,
                                "produce_iso": _iso_local(now_ts),
                                "stream_url": stream_url,
                                "stream_index": stream_index,
                                "stream_segment_index": stream_segment_index,
                                "poll_round": poll_round,
                            })
                            if not ok:
                                return
                            seq += 1
                        if usage_part:
                            usage = usage_part
                    full_text = "".join(buf)
                else:
                    full_text, usage = nonstream_pair or ("", None)  # type: ignore

                latency_ms = int((time.time() - t_start) * 1000)
                now_ts = time.time()
                ok = _q_put_with_retry({
                    "type": "asr_stream_done",
                    "segment_index": seg_idx,
                    "full_text": full_text,
                    "usage": {"vad": vad_info, **(usage or {})},
                    "model": asr_model_name,
                    "latency_ms": latency_ms,
                    "lang": lang,
                    "sr_hz": sr_hz,
                    "t0": t0, "t1": t1,
                    "produce_ts": now_ts,
                    "produce_iso": _iso_local(now_ts),
                    "stream_url": stream_url,
                    "stream_index": stream_index,
                    "stream_segment_index": stream_segment_index,
                    "poll_round": poll_round,
                })
                if not ok:
                    return

                logger.info("[C] ASR 完成 seg#%s", seg_idx)

            finally:
                try:
                    q_audio.task_done()
                except Exception:
                    pass

    except Exception as e:
        logger.exception(f"[C] 异常退出：{e}")
    finally:
        logger.info("[C] 线程退出清理完成。")
