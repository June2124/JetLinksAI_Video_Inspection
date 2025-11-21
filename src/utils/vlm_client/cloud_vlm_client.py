'''
Author: 13594053100@163.com
Date: 2025-10-27 13:49:42
LastEditTime: 2025-10-27 16:27:30
'''
from dashscope import MultiModalConversation
import time

from typing import Dict, Optional, Tuple,Iterator
from src.utils import logger_utils
from queue import Queue, Empty

logger = logger_utils.get_logger(__name__)

class CloudVLMClient:
    """
    云端多模态大模型推理客户端。
    - 单例
    - 对外暴露 infer()
    """
    def __init__(
    self,
    model_name: str,
    messages: list[Dict],
    *,
    want_streaming: bool,
    open_timeout_s: float = 15.0,
    nonstream_max_tries: int = 2,
    nonstream_backoff_base: float = 1.6,
    response_format: Optional[dict] = None,  # 任务型：json_schema；描述型：None
    HAS_MMC: bool = True,
    q_ctrl: Optional[Queue] = None,
    stop: Optional[object] = None,
    ):
        # TODO 默认不校验参数，上游保证正确性, 后续可加参数校验
        self.model_name = model_name
        self.messages = messages
        self.want_streaming = want_streaming
        self._on_heartbeat = (lambda: _poll_ctrl_heartbeat(q_ctrl, stop)) if q_ctrl and stop else None
        self.open_timeout_s = open_timeout_s
        self.nonstream_max_tries = nonstream_max_tries
        self.nonstream_backoff_base = nonstream_backoff_base
        self.response_format = response_format
        self._HAS_MMC = HAS_MMC
        self.q_ctrl = q_ctrl
        self.stop = stop
    
    def infer(self) -> Tuple[str, Optional[Iterator[Tuple[Optional[str], Optional[dict]]]], Optional[Tuple[str, Optional[dict]]]]:
        """
        对外接口:
        - 调用云端API进行推理
        - 返回完整文本（可JSON）
        """

        """
        - want_streaming=True：先试流式，失败则本次回退到非流式
        - want_streaming=False：直接非流式
        返回：
        - 若流式成功：("stream", iterator, None)
        - 若非流式：("nonstream", None, (full_text, usage))
        """
        if not self._HAS_MMC:
            raise RuntimeError("DashScope SDK 不可用")

        # 尝试流式（仅用于描述型）
        if self.want_streaming:
            try:
                def _open_stream_call():
                    t0 = time.time()
                    while True:
                        try:
                            return MultiModalConversation.call(
                                model=self.model_name,
                                messages=self.messages,
                                stream=True,
                                incremental_output=True,
                            )
                        except Exception as e:
                            if time.time() - t0 > self.open_timeout_s:
                                raise e
                            time.sleep(0.2)

                stream_obj = _open_stream_call()

                def _iter() -> Iterator[Tuple[Optional[str], Optional[dict]]]:
                    usage_cache: Optional[dict] = None
                    for rsp in stream_obj:
                        if callable(self._on_heartbeat) and self._on_heartbeat():
                            break
                        delta, usage_part = None, None
                        try:
                            out = rsp.get("output") or {}
                            choices = out.get("choices") or []
                            if choices:
                                msg = (choices[0] or {}).get("message") or {}
                                content = msg.get("content") or []
                                parts = [it.get("text") for it in content if isinstance(it, dict) and it.get("text")]
                                if parts:
                                    delta = "".join(parts)
                            u = rsp.get("usage") or {}
                            if u:
                                usage_cache = {
                                    "prompt_tokens": u.get("input_tokens") or u.get("prompt_tokens"),
                                    "completion_tokens": u.get("output_tokens") or u.get("completion_tokens"),
                                    "total_tokens": u.get("total_tokens"),
                                }
                                usage_part = usage_cache
                        except Exception:
                            pass
                        yield delta, usage_part

                return "stream", _iter(), None
            except Exception as e:
                logger.warning("[B] 流式失败，本次回退非流式：%s", e)

        # 非流式（或回退）：直接传 response_format
        last_err = None
        for i in range(1, self.nonstream_max_tries + 1):
            try:
                resp = MultiModalConversation.call(
                    model=self.model_name,
                    messages=self.messages,
                    stream=False,
                    response_format=self.response_format if self.response_format else None,
                )
                ft, usage = _parse_text_and_usage_from_resp(resp)
                return "nonstream", None, (ft, usage)
            except Exception as e:
                last_err = e
                if i < self.nonstream_max_tries:
                    sleep_s = min(self.nonstream_backoff_base ** i, 10.0) + min(0.6, 0.2 * i)
                    logger.warning("[B] 非流式失败 %d/%d，%.1fs后重试：%s", i, self.nonstream_max_tries, sleep_s, e)
                    time.sleep(sleep_s)
        raise last_err

# ----------------- DashScope Streaming/Non-Streaming 辅助 -----------------
def _parse_text_and_usage_from_resp(resp: dict) -> Tuple[str, Optional[dict]]:
    """
    从 DashScope 响应中解析文本内容和用量信息。

    Args:
        resp (dict): DashScope API 的响应字典。

    Returns:
        Tuple[str, Optional[dict]]: 返回解析出的文本内容和用量信息（usage）。
    """
    out = (resp or {}).get("output") or {}
    usage = (resp or {}).get("usage") or None
    text = ""
    try:
        choices = out.get("choices") or []
        if choices:
            msg = (choices[0] or {}).get("message") or {}
            content = msg.get("content") or []
            parts = [it.get("text") for it in content if isinstance(it, dict) and it.get("text")]
            text = "".join(parts) if parts else ""
    except Exception:
        pass
    return text, usage

def _poll_ctrl_heartbeat(q_ctrl: Queue, stop: object) -> bool:
        try:
            msg = q_ctrl.get_nowait()
        except Empty:
            return False
        if msg is stop:
            return True
        if isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN"):
            return True
        try:
            q_ctrl.put_nowait(msg)
        except Exception:
            pass
        return False
        

