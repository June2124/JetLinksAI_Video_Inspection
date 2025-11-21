'''
Author: 13594053100@163.com
Date: 2025-10-24 15:47:27
LastEditTime: 2025-11-18 14:08:35
'''

import os
from typing import Optional, Literal
from pydantic import BaseModel, field_validator,ConfigDict
from src.all_enum import  CLOUD_VLM_MODEL_NAME, LOCAL_VLM_MODEL_NAME, VLM_DETECT_EVENT_LEVEL

CURRENT_FILE_PATH = os.path.realpath(__file__)
BASE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(CURRENT_FILE_PATH)))
BASE_PATH_ABSPATH = os.path.abspath(BASE_PATH)

class VlmConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    offline_system_prompt: Optional[str] = "" # offline离线音视频解析的vlm侧系统提示词只能由上层传入, 且没有预设
    vlm_cloud_model_name: CLOUD_VLM_MODEL_NAME = CLOUD_VLM_MODEL_NAME.QWEN3_VL_PLUS
    vlm_local_model_name: LOCAL_VLM_MODEL_NAME = LOCAL_VLM_MODEL_NAME.QWEN3_VL_8B
    vlm_streaming: bool = False

    # 选择VLM后端 cloud=走DashScope，local=走盒子本地model
    vlm_backend: Literal["cloud","local"] = "local"

    # 历史上下文/事件筛选等原有配置
    vlm_task_history_enabled: bool = False
    vlm_event_min_level: VLM_DETECT_EVENT_LEVEL = VLM_DETECT_EVENT_LEVEL.LOW

    # 证据帧静态导出
    vlm_static_evidence_images_dir: str = os.path.join(BASE_PATH_ABSPATH,"static","evidence_images")
    vlm_static_evidence_images_url_prefix: str = "/static/evidence_images"

    @field_validator("offline_system_prompt")
    @classmethod
    def check_length(cls, v):
        if isinstance(v, str) and len(v) > 300:
            raise ValueError("系统提示词需少于300字")
        return v

    
