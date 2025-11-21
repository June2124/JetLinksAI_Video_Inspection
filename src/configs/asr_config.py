'''
Author: 13594053100@163.com
Date: 2025-10-22 10:22:45
LastEditTime: 2025-11-16 00:58:17
'''

from typing import Literal, Optional

from pydantic import BaseModel, Field, ConfigDict, model_validator

from src.all_enum import (
    CLOUD_ASR_MODEL_NAME,
    LOCAL_ASR_MODEL_NAME,
)


class AsrConfig(BaseModel):
    model_config = ConfigDict(validate_assignment=True)

    # 这些参数仅在 cloud 模式下生效，local 模式下不允许设置
    disfluency_removal_enabled: Optional[bool] = Field(
        default=None, description="是否开启过滤语气词（仅云端 ASR 模型支持）"
    )
    semantic_punctuation_enabled: Optional[bool] = Field(
        default=None, description="语义断句开关（仅云端 ASR 模型支持）"
    )
    max_sentence_silence: Optional[int] = Field(
        default=None,
        ge=200,
        le=6000,
        description="VAD 断句静音时长阈值，单位 ms（仅云端 ASR 模型支持）",
    )
    punctuation_prediction_enabled: Optional[bool] = Field(
        default=None, description="自动加标点（仅云端 ASR 模型支持）"
    )
    inverse_text_normalization_enabled: Optional[bool] = Field(
        default=None, description="逆文本正则化（仅云端 ASR 模型支持）"
    )

    asr_cloud_model_name: CLOUD_ASR_MODEL_NAME = CLOUD_ASR_MODEL_NAME.PARAFORMER_REALTIME_V2
    asr_local_model_name: LOCAL_ASR_MODEL_NAME = LOCAL_ASR_MODEL_NAME.WHISPER

    # 选择当前使用云端还是本地 ASR
    asr_backend: Literal["cloud", "local"] = "cloud"

    @model_validator(mode="after")
    def check_parameter(self) -> "AsrConfig":
        """local 模式下，禁止设置云端才支持的参数"""
        if self.asr_backend == "local":
            forbidden_fields = {
                "disfluency_removal_enabled": self.disfluency_removal_enabled,
                "semantic_punctuation_enabled": self.semantic_punctuation_enabled,
                "max_sentence_silence": self.max_sentence_silence,
                "punctuation_prediction_enabled": self.punctuation_prediction_enabled,
                "inverse_text_normalization_enabled": self.inverse_text_normalization_enabled,
            }
            # 只要有任意一个字段被设置（非 None），就报错
            invalid = [name for name, value in forbidden_fields.items() if value is not None]
            if invalid:
                # 拼一个清晰的错误提示，把具体字段名字列出来
                fields_str = "、".join(invalid)
                raise ValueError(
                    f"本地音频转录模型(asr_backend='local')不支持以下参数设置：{fields_str}。"
                    f"请删除这些字段或将 asr_backend 切换为 'cloud'。"
                )
        return self
