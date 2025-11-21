'''
Author: 13594053100@163.com
Date: 2025-11-18 10:26:58
LastEditTime: 2025-11-19 10:36:55
'''
from pydantic import BaseModel

class RuntimeMachineConfig(BaseModel):
    # 是否开启本地VLM运行时状态机
    local_vlm_runtime_machine: bool = True # 开启后, 状态机会在本地VLM服务器延迟过高时, 通过自适应算法动态修改切窗大小和轮询间隔; 关闭时则固定使用初始参数.
    # 是否开启运行时队列状态机
    queue_runtime_machine: bool = True 
    
