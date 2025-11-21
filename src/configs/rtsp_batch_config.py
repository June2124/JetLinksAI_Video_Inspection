'''
Author: 13594053100@163.com
Date: 2025-11-04 09:36:22
LastEditTime: 2025-11-12 20:38:16
'''
from typing import Optional, List, Union
from pydantic import BaseModel, Field, field_validator, model_validator, ConfigDict
from src.all_enum import VLM_SYSTEM_PROMPT_PRESET
from src.configs.vlm_config import VlmConfig
from src.utils.logger_utils import get_logger

logger = get_logger(__name__)

class RTSP(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    rtsp_url: Optional[str] = None
    rtsp_system_prompt: Optional[Union[VLM_SYSTEM_PROMPT_PRESET, str]] = None
    rtsp_cut_number: int = Field(ge=1, le=20, default=1, description="该流在一轮轮询中要切几个窗口")

    @field_validator("rtsp_system_prompt")
    @classmethod
    def _limit_len(cls, v):
        if isinstance(v, str) and len(v) > 300:
            raise ValueError("rtsp_system_prompt 文本需少于300字")
        return v

class RTSPBatchConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    polling_list: List[RTSP]
    polling_batch_interval: float = Field(
        default=60.0 * 10.0, ge=10.0, description="多流轮询时，两轮轮询之间的间隔，单位秒"
    )
    vlm_config: VlmConfig

    @model_validator(mode="after")
    def _check_vlm_config(self):
        if self.vlm_config.vlm_streaming:
            raise ValueError("SECURITY_SINGLE/SECURITY_POLLING 模式下不允许流式输出")
        if self.vlm_config.offline_system_prompt:
            logger.warning(
                "SECURITY_SINGLE/SECURITY_POLLING 模式下, VLM 的系统提示词须由 rtsp_system_prompt 定义，"
                "offline_system_prompt 将被忽略"
            )
        return self
