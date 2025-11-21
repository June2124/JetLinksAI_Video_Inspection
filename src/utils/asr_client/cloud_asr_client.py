from __future__ import annotations
'''
Author: 13594053100@163.com
Date: 2025-11-12 16:30:50
LastEditTime: 2025-11-12 16:30:54
'''
import os
import wave
from typing import Optional, Dict, Any, Iterable, List, Tuple, Generator

from src.utils import logger_utils

logger = logger_utils.get_logger(__name__)

# ---- DashScope Paraformer SDK ----
try:
    import dashscope  # type: ignore
    from dashscope.audio.asr import Recognition, RecognitionCallback  # type: ignore
    _HAVE_PARA = True
except Exception as e:
    _HAVE_PARA = False
    _PARA_IMPORT_ERR = e


def _is_http_url(p: str) -> bool:
    return isinstance(p, str) and p.lower().startswith(("http://", "https://"))

def _to_local_path(maybe_uri: str) -> str:
    """
    接受 文件路径 或 file://URI，返回本地路径。
    不支持 http/https（ASR 这里只读本地 WAV）
    """
    if not maybe_uri:
        raise ValueError("audio_uri is empty")
    if _is_http_url(maybe_uri):
        raise ValueError("CloudASRClient expects local WAV file, got http/https url")
    if maybe_uri.lower().startswith("file://"):
        return maybe_uri[len("file://"):]
    return maybe_uri

def _iter_pcm_frames_from_wav(wav_path: str, *, block_bytes: int = 3200) -> Iterable[bytes]:
    """
    读取 16kHz/mono/PCM16 WAV，按 100ms(3200B) 切块产出纯 PCM16。
    """
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


class _ParaCallback(RecognitionCallback):
    """
    仅收集 sentence_end=True 的句级结果。
    """
    def __init__(self):
        super().__init__()
        import threading
        self._lock = threading.Lock()
        self._done = threading.Event()
        self._err: Optional[str] = None
        self.sentences: List[dict] = []

    # SDK 结束
    def on_complete(self) -> None:
        self._done.set()

    # SDK 错误
    def on_error(self, message) -> None:
        try:
            req_id = getattr(message, "request_id", None)
            msg = getattr(message, "message", None)
            logger.error(f"[CloudASRClient] Paraformer 错误: request_id={req_id}, err={msg}")
        except Exception:
            logger.error("[CloudASRClient] Paraformer 错误（无法解析 message）")
        self._err = "error"
        self._done.set()

    # SDK 事件（持续回调）
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


class CloudASRClient:
    """
    适配 C 线程(worker_c_asr) 所需的统一接口。

    参数
    ----
    model_name : str
        Paraformer 模型名（如 "paraformer-realtime-v2"）
    audio_uri : str
        音频路径或 file://URI（需要 16k/mono/PCM16 WAV）
    want_streaming : bool
        True: 以“流式模式”返回（yield 句级 delta）；
        False: 以“非流式模式”一次性返回整段文本。
        注意：Paraformer 句级结果在 stop() 后才能完整产出，
        因此“流式”也会在 stop() 之后一次性 yield 多个句子。
    q_ctrl / stop :
        可选的控制队列/哨兵，若不需要可传 None。
    asr_options : dict
        透传 SDK 可选项（语义标点、去填充词、句末静音阈值等）：
        {
            "semantic_punctuation_enabled": bool,
            "disfluency_removal_enabled": bool,
            "max_sentence_silence": int,
            "punctuation_prediction_enabled": bool,
            "inverse_text_normalization_enabled": bool,
            "sample_rate": 16000,
            "block_bytes": 3200,
            "wait_done_seconds": 10.0
        }

    返回
    ----
    mode, iter_pair, nonstream_pair
      - mode == "stream":  iter_pair 为一个生成器，yield (delta:str, usage_part:dict|None)
      - mode == "nonstream": nonstream_pair 为 (full_text:str, usage:dict|None)
    """
    def __init__(
        self,
        *,
        model_name: str,
        audio_uri: str,
        want_streaming: bool = True,
        q_ctrl=None,
        stop=None,
        asr_options: Optional[Dict[str, Any]] = None,
    ):
        if not _HAVE_PARA:
            raise RuntimeError(f"未安装/导入 Paraformer SDK（dashscope），error={_PARA_IMPORT_ERR}")

        self.model_name = model_name or os.getenv("ASR_MODEL", "paraformer-realtime-v2")
        self.audio_path = _to_local_path(audio_uri)
        self.want_streaming = bool(want_streaming)
        self.q_ctrl = q_ctrl
        self.stop = stop

        self.opt = {
            "semantic_punctuation_enabled": False,
            "disfluency_removal_enabled": False,
            "max_sentence_silence": 500,
            "punctuation_prediction_enabled": True,
            "inverse_text_normalization_enabled": True,
            "sample_rate": 16000,
            "block_bytes": 3200,          # 100ms (PCM16 @16k)
            "wait_done_seconds": 10.0,
        }
        if asr_options:
            self.opt.update(asr_options)

        # API Key
        api_key = os.environ.get("DASHSCOPE_API_KEY")
        if api_key:
            dashscope.api_key = api_key

    # 对外主入口
    def infer(self) -> Tuple[str, Optional[Generator[Tuple[str, Optional[dict]], None, None]], Optional[Tuple[str, Optional[dict]]]]:
        """
        若 want_streaming=True  -> 返回 ("stream", generator, None)
        若 want_streaming=False -> 返回 ("nonstream", None, (full_text, usage))
        """
        if self.want_streaming:
            return "stream", self._iter_sentences_stream(), None
        else:
            text, usage = self._run_and_collect_all()
            return "nonstream", None, (text, usage)

    # ===== 内部：生成器，按“句子”yield delta =====
    def _iter_sentences_stream(self) -> Generator[Tuple[str, Optional[dict]], None, None]:
        sentences = self._run_paraformer_and_get_sentences()
        # usage 可选择在每句都给 None；统一在段尾由 worker 汇总
        for s in sentences:
            text = (s.get("text") or "").strip()
            if not text:
                continue
            yield text, None   # (delta, usage_part)

    # ===== 内部：一次性跑完，合并全文 =====
    def _run_and_collect_all(self) -> Tuple[str, Optional[dict]]:
        sentences = self._run_paraformer_and_get_sentences()
        full_text = "".join([(s.get("text") or "") for s in sentences])
        # usage 简化：SDK 暂无稳定 usage，返回 None；由上层 VAD/统计补充
        return full_text, None

    # ===== 实际调用 SDK，返回 sentence_end=True 的句对象列表 =====
    def _run_paraformer_and_get_sentences(self) -> List[dict]:
        # WAV 参数校验在读取器中做
        cb = _ParaCallback()
        recog = Recognition(
            model=self.model_name,
            format='pcm',
            sample_rate=int(self.opt["sample_rate"]),
            semantic_punctuation_enabled=bool(self.opt["semantic_punctuation_enabled"]),
            disfluency_removal_enabled=bool(self.opt["disfluency_removal_enabled"]),
            max_sentence_silence=int(self.opt["max_sentence_silence"]),
            punctuation_prediction_enabled=bool(self.opt["punctuation_prediction_enabled"]),
            inverse_text_normalization_enabled=bool(self.opt["inverse_text_normalization_enabled"]),
            callback=cb
        )

        # 发送音频
        try:
            recog.start()
            for pcm in _iter_pcm_frames_from_wav(self.audio_path, block_bytes=int(self.opt["block_bytes"])):
                # 控制中断（可选）
                if self._check_stop():
                    logger.info("[CloudASRClient] 检测到 STOP，提前终止发送音频")
                    break
                recog.send_audio_frame(pcm)
            recog.stop()
        except Exception as e:
            logger.warning(f"[CloudASRClient] Paraformer 发送/停止异常：{e}")

        # 等待完成并取结果
        cb.wait_done(timeout=float(self.opt["wait_done_seconds"]))
        if cb.has_error():
            logger.warning("[CloudASRClient] Paraformer 识别报错（已记录），将返回已收集的句级结果。")

        sentences = cb.fetch_sentences()  # 仅 sentence_end=True
        return sentences

    # 控制检测：从 q_ctrl 非阻塞 peek 一下 STOP
    def _check_stop(self) -> bool:
        if self.q_ctrl is None or self.stop is None:
            return False
        try:
            msg = self.q_ctrl.get_nowait()
        except Exception:
            return False
        try:
            if (msg is self.stop) or (isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN")):
                return True
        finally:
            # 非 STOP 的消息放回
            try:
                self.q_ctrl.put_nowait(msg)
            except Exception:
                pass
        return False
