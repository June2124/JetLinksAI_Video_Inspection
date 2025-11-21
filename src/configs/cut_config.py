'''
Author: 13594053100@163.com
Date: 2025-10-22 14:26:38
LastEditTime: 2025-11-18 15:16:51
'''

from pydantic import BaseModel, Field,ConfigDict


class CutConfig(BaseModel):
   model_config = ConfigDict(validate_assignment=True)
   
   cut_window_sec: float = Field(default=4.0, ge=1.0, le=60.0, description="每个切片的时长，单位秒")
   alpha_bgr: float = Field(default=0.5, ge=0.0, le=1.0, description="背景融合系数")
   topk_frames: int = Field(default=1, ge=1, description="每个切片保留的关键帧数量")
   
   
   
