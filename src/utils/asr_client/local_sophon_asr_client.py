from __future__ import annotations
'''
Author: 13594053100@163.com
Date: 2025-11-12 16:32:39
LastEditTime: 2025-11-12 16:32:43
'''
from typing import Optional, Dict, Any, Tuple, Generator

from src.utils import logger_utils
logger = logger_utils.get_logger(__name__)

class LocalASRClient:
    """
    本地 ASR 占位客户端。
    接口对齐 CloudASRClient：
      infer() -> (mode, iter_pair, nonstream_pair)
        - mode == "stream":  iter_pair 为生成器，yield (delta:str, usage_part:dict|None)
        - mode == "nonstream": nonstream_pair 为 (full_text:str, usage:dict|None)
    """
    def __init__(self, *, model_name: str, audio_uri: str, q_ctrl=None, stop=None, asr_options: Optional[Dict[str, Any]] = None):
        self.model_name = model_name
        self.audio_uri = audio_uri
        self.q_ctrl = q_ctrl
        self.stop = stop
        self.opt = asr_options or {}

    def infer(self) -> Tuple[str, Optional[Generator[Tuple[str, Optional[dict]], None, None]], Optional[Tuple[str, Optional[dict]]]]:
        raise NotImplementedError("LocalASRClient 尚未实现本地 ASR 推理。")
