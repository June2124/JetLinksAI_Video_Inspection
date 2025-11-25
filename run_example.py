'''
Author: 13594053100@163.com
Date: 2025-10-24 15:46:46
LastEditTime: 2025-11-25 09:45:56
'''

from JetLinksAI.streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL, LOCAL_VLM_MODEL_NAME, VLM_SYSTEM_PROMPT_PRESET
from src.configs.cut_config import CutConfig
from src.configs.vlm_config import VlmConfig
from src.configs.rtsp_batch_config import RTSPBatchConfig, RTSP
from src.configs.runtime_machine_config import RuntimeMachineConfig

# 1.离线视频
mode = MODEL.OFFLINE
url = "file://video_path"
enable_b = True # B线程是视频/RTSP流安防线程，需开启该线程
enable_c = False # C线程是音频转录线程，需关闭该线程

rtsp_batch_config = None # 当模式为离线视频解析, 是不允许传入此参数的, 该参数默认值就是None
cut_config = CutConfig(cut_window_sec=4, alpha_bgr=0.5, topk_frames=1) # 基本切窗配置, 目前必须将topk_frames设置为1, 否则调用本地模型推理服务时会报错

offline_system_prompt = "描述你看到的画面" # 注意：当mode为 OFFLINE 时, 提示词只能由层传入，没有传入则默认没有提示词由模型自由发挥
vlm_config = VlmConfig(offline_system_prompt=offline_system_prompt, vlm_local_model_name=LOCAL_VLM_MODEL_NAME.QWEN3_VL_8B, vlm_backend="local") # 其他的配置项直接走默认

runtime_machine_config = RuntimeMachineConfig(local_vlm_runtime_machine=True) # 默认本地VLM运行时状态机为开启, 可以设置为不开启

sa = StreamingAnalyze(mode=mode, url=url, enable_b=enable_b, enable_c=enable_c, rtsp_batch_config=rtsp_batch_config,
                      cut_config=cut_config, vlm_config=vlm_config, runtime_machine_config=runtime_machine_config)

try:
    for item in sa.run():
        print(item) # 会将每次分析结果字典透传出来, 上层应按需做数据后处理
        """
        payload = {
        "type": "vlm_stream_done",
        "segment_index": seg_idx,
        "full_text": full_text or "",
        "usage": usage,
        "model": model,
        "streaming": bool(streaming),
        "latency_ms": latency_ms,
        "produce_ts": time.time(),
        # 时间轴/帧
        "clip_t0": item.get("t0"),
        "clip_t1": item.get("t1"),
        "frame_pts": item.get("frame_pts") or [],
        "frame_indices": item.get("frame_indices") or [],
        "t0_iso": item.get("t0_iso"),
        "t1_iso": item.get("t1_iso"),
        "t0_epoch": item.get("t0_epoch"),
        "t1_epoch": item.get("t1_epoch"),
        "frame_epoch": item.get("frame_epoch") or [],
        "frame_iso": item.get("frame_iso") or [],
        # 透传去重/上下文/证据
        "suppressed_dup": bool(suppressed_dup) if suppressed_dup is not None else None,
        "ctx_rounds": ctx_rounds,
        "evidence_images": evidence_images or [],
        "evidence_image_urls": evidence_image_urls or [],
        # 编码策略
        "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
        "origin_policy": (item.get("policy") or {}).get("policy_used"),
        # 透传流元信息
        "stream_url": item.get("stream_url"),            # 当前片段所属的 RTSP 地址（唯一标识一条流）
        "stream_index": item.get("stream_index"),        # 当前流在轮询列表中的索引位置（0-based）
        "stream_segment_index": item.get("stream_segment_index"),  # 该流自任务启动以来的连续窗口序号（全局计数）
        "window_index_in_stream": item.get("window_index_in_stream"), # 该流在“当前轮”中是第几个窗口（局部计数）
        "polling_round_index": item.get("polling_round_index"),    # 当前是第几轮轮询（SECURITY_POLLING 模式下从 0 开始计数）
    }
        """
except Exception as e:
    raise e

# 2.单流RTSP常驻安防

mode = MODEL.SECURITY_SINGLE
enable_b = True # B线程是视频/RTSP流安防线程，需开启该线程
enable_c = False # C线程是音频转录线程，需关闭该线程

vlm_config = VlmConfig(vlm_local_model_name=LOCAL_VLM_MODEL_NAME.QWEN3_VL_8B, vlm_backend="local") # 当模式为MODEL.SECURITY_SINGLE或MODEL.SECURITY_POLLING时不允许将rtsp的提示词
                                                                                                    # 放在offline_system_prompt中配置

# 每个RTSP流就是一个RTSP对象, 需要先构造该对象
url = "rtsp/rtsps:url"
rtsp_system_prompt = VLM_SYSTEM_PROMPT_PRESET.OFFICE_PRODUCTIVITY_COMPLIANT # 使用预设提示词, 保证一定返回JSON数据; 上层也可以直接传入字符串类型的系统提示词, 不保证结构化返回
rtsp_cut_number = 1 # 该参数在单流常驻模式MODEL.SECURITY_SINGLE下无效, 这里仅为了保证配置统一
rtsp1 = RTSP(url=url, rtsp_system_prompt=rtsp_system_prompt, rtsp_cut_number=rtsp_cut_number) 

polling_list = []
polling_list.append(rtsp1) # 单流常驻只能向rtsp流列表里添加一个流, 多余的流会在初始化StreamingAnalyze阶段被舍弃
rtsp_batch_config = RTSPBatchConfig(polling_list=polling_list, polling_batch_interval=60, vlm_config=vlm_config) # 此处传入vlm_config参数的目的是为了pydantic做跨数据模型属性检验
cut_config = CutConfig(cut_window_sec=4, alpha_bgr=0.5, topk_frames=1) # 基本切窗配置, 目前必须将topk_frames设置为1, 否则调用本地模型推理服务时会报错

runtime_machine_config = RuntimeMachineConfig(local_vlm_runtime_machine=True) # 默认本地VLM运行时状态机为开启, 可以设置为不开启

sa = StreamingAnalyze(mode=mode, enable_b=enable_b, enable_c=enable_c, rtsp_batch_config=rtsp_batch_config,
                      cut_config=cut_config, vlm_config=vlm_config, runtime_machine_config=runtime_machine_config) # 当模式为MODEL.SECURITY_SINGLE或MODEL.SECURITY_POLLING时不允许将rtsp的流地址
                                                                                                                    # 放在url里面配置

try:
    for item in sa.run():
        print(item) # 会将每次分析结果字典透传出来, 上层应按需做数据后处理
        """
        payload = {
        "type": "vlm_stream_done",
        "segment_index": seg_idx,
        "full_text": full_text or "",
        "usage": usage,
        "model": model,
        "streaming": bool(streaming),
        "latency_ms": latency_ms,
        "produce_ts": time.time(),
        # 时间轴/帧
        "clip_t0": item.get("t0"),
        "clip_t1": item.get("t1"),
        "frame_pts": item.get("frame_pts") or [],
        "frame_indices": item.get("frame_indices") or [],
        "t0_iso": item.get("t0_iso"),
        "t1_iso": item.get("t1_iso"),
        "t0_epoch": item.get("t0_epoch"),
        "t1_epoch": item.get("t1_epoch"),
        "frame_epoch": item.get("frame_epoch") or [],
        "frame_iso": item.get("frame_iso") or [],
        # 透传去重/上下文/证据
        "suppressed_dup": bool(suppressed_dup) if suppressed_dup is not None else None,
        "ctx_rounds": ctx_rounds,
        "evidence_images": evidence_images or [],
        "evidence_image_urls": evidence_image_urls or [],
        # 编码策略
        "small_video_fps": item.get("policy", {}).get("encode", {}).get("fps"),
        "origin_policy": (item.get("policy") or {}).get("policy_used"),
        # 透传流元信息
        "stream_url": item.get("stream_url"),            # 当前片段所属的 RTSP 地址（唯一标识一条流）
        "stream_index": item.get("stream_index"),        # 当前流在轮询列表中的索引位置（0-based）
        "stream_segment_index": item.get("stream_segment_index"),  # 该流自任务启动以来的连续窗口序号（全局计数）
        "window_index_in_stream": item.get("window_index_in_stream"), # 该流在“当前轮”中是第几个窗口（局部计数）
        "polling_round_index": item.get("polling_round_index"),    # 当前是第几轮轮询（SECURITY_POLLING 模式下从 0 开始计数）
    }
        """
except Exception as e:
    raise e

# 当你需要在运行时修改基本切窗大小cut_window_sec, 调用update_cut_window_sec()方法. 该方法只能在模式为MODEL.SECURITY_SINGLE或MODEL.SECURITY_POLLING时被调用
sa.update_cut_window_sec(new_cut_window_sec=6.5)

# 当你需要在运行时替换某一个正在运行的流
current_rtsp_url = url

new_rtsp_url = "new rtsp/rtsps:url"
new_rtsp_system_prompt = VLM_SYSTEM_PROMPT_PRESET.HEALTHCARE_PUBLIC_SAFETY # 你也可以传入自定义字符串
new_rtsp_cut_number = 1 # 该参数在单流常驻模式MODEL.SECURITY_SINGLE下无效, 这里仅为了保证配置统一
rtsp2 = RTSP(url=new_rtsp_url, rtsp_system_prompt=new_rtsp_system_prompt, rtsp_cut_number=new_rtsp_cut_number)

sa.update_stream_rtsp(current_rtsp_url=current_rtsp_url, new_rtsp=rtsp2)

# 当你需要暂停安防任务时, 调用pauseI()方法
sa.pause()

# 当你需要从暂停态重启安防流程时, 调用resume()方法. 重启任务后. 会自动跳到当前直播点
sa.resume()

# 当你需要在任意时刻停止安放流程, 调用force_stop()方法
sa.force_stop()

# 3.多流轮询RTSP安防
mode = MODEL.SECURITY_POLLING
enable_b = True   # B线程是视频/RTSP流安防线程，需开启该线程
enable_c = False  # C线程是音频转录线程，轮询安防一般不需要语音转写

# 轮询模式下，VLM 的系统提示词须由各 RTSP 对象的 rtsp_system_prompt 定义
vlm_config = VlmConfig(
    vlm_local_model_name=LOCAL_VLM_MODEL_NAME.QWEN3_VL_8B,
    vlm_backend="local"   # 轮询安防通常走本地模型，降低时延和外网依赖
)

# === 3.1 构造多路 RTSP 流配置 ===
# 注意：RTSP 数据类字段名为 rtsp_url / rtsp_system_prompt / rtsp_cut_number
#      当模式为 SECURITY_SINGLE / SECURITY_POLLING 时，不允许把流地址放在 StreamingAnalyze.__init__ 的 url 参数中

rtsp_url_1 = "rtsp://camera-1/live"
rtsp_url_2 = "rtsp://camera-2/live"

# 使用预设提示词，保证尽量返回结构化 JSON；也可以直接传入自定义字符串提示词（不保证严格结构化）
rtsp1 = RTSP(
    rtsp_url=rtsp_url_1,
    rtsp_system_prompt=VLM_SYSTEM_PROMPT_PRESET.FACTORY_SECURITY,
    rtsp_cut_number=1  # 一轮轮询中从该流切 1 个窗口
)

rtsp2 = RTSP(
    rtsp_url=rtsp_url_2,
    rtsp_system_prompt=VLM_SYSTEM_PROMPT_PRESET.OFFICE_PRODUCTIVITY_COMPLIANT,
    rtsp_cut_number=2  # 一轮轮询中从该流切 2 个窗口（比如重点关注的监控点）
)

polling_list = [rtsp1, rtsp2]

# polling_batch_interval：两轮轮询之间的间隔（秒），SECURITY_POLLING 下才生效
rtsp_batch_config = RTSPBatchConfig(
    polling_list=polling_list,
    polling_batch_interval=60.0,  # 每 60s 完成一轮轮询，再等待 60s 进入下一轮
    vlm_config=vlm_config          # 传入用于跨模型校验
)

# 基本切窗配置：目前 topk_frames 必须为 1，否则调用本地 VLM 推理服务会报错
cut_config = CutConfig(
    cut_window_sec=4.0,
    alpha_bgr=0.5,
    topk_frames=1
)

# 默认本地 VLM 运行时状态机为开启，可以设置为 False 关闭
runtime_machine_config = RuntimeMachineConfig(
    local_vlm_runtime_machine=True
)

# 初始化多流轮询安防任务
sa = StreamingAnalyze(
    mode=mode,
    enable_b=enable_b,
    enable_c=enable_c,
    rtsp_batch_config=rtsp_batch_config,  # SECURITY_POLLING 模式必须通过这个参数传入多路流配置
    cut_config=cut_config,
    vlm_config=vlm_config,
    runtime_machine_config=runtime_machine_config
)

try:
    for i, item in enumerate(sa.run(), start=1):
        # 上层可按需做后处理；这里直接打印原始结果
        print(item)

        """
        item 中的关键字段说明（与前两个示例一致）：
        {
            "t0": ..., "t1": ...,
            "t0_iso": ..., "t1_iso": ...,
            "frame_pts": [...],
            "frame_indices": [...],
            "frame_iso": [...],
            "frame_epoch": [...],
            "policy": {...},
            "stream_url": item.get("stream_url"),                    # 当前片段所属的 RTSP 地址
            "stream_index": item.get("stream_index"),                # 在 polling_list 中的索引（0-based）
            "stream_segment_index": item.get("stream_segment_index"),# 该流自任务启动以来的连续窗口序号
            "window_index_in_stream": item.get("window_index_in_stream"), # 当前轮中该流第几个窗口
            "polling_round_index": item.get("polling_round_index"),  # 当前是第几轮轮询（从 0 开始）
            ...
        }
        """

        # 多流轮询也可以使用: update_cut_window_sec() 、update_stream_rtsp() 、 pause() 、 resume() 、 force_stop()方法; 以下为轮询专属方法：
        # ========= 3.2 演示轮询模式专属控制方法 =========

        # ① 动态调整两轮轮询间隔：update_polling_batch_interval
        #    假设跑到第 10 个窗口后，发现整体负载较低，可以把轮询间隔从 60s 提高到 120s
        if i == 10:
            sa.update_polling_batch_interval(new_interval=120.0)
            print("[CONTROL] 已将 polling_batch_interval 更新为 120 秒")

        # ② 动态移除某一路 RTSP：polling_remove_stream
        #    比如在第 20 个窗口时，不再关注第二个摄像头
        if i == 20:
            # 可以通过 index 或 rtsp_url 二选一，这里演示按 URL 删除
            sa.polling_remove_stream(rtsp_url=rtsp_url_2)
            print(f"[CONTROL] 已从轮询列表移除流: {rtsp_url_2}")

        # ③ 动态新增某一路 RTSP：polling_add_stream
        #    在第 30 个窗口时，接入第三个摄像头
        if i == 30:
            rtsp_url_3 = "rtsp://camera-3/live"
            rtsp3 = RTSP(
                rtsp_url=rtsp_url_3,
                rtsp_system_prompt=VLM_SYSTEM_PROMPT_PRESET.HEALTHCARE_PUBLIC_SAFETY,
                rtsp_cut_number=1
            )
            sa.polling_add_stream(add_rtsp=rtsp3)
            print(f"[CONTROL] 已向轮询列表新增流: {rtsp_url_3}")

except Exception as e:
    # 上层按需处理异常（重启任务 / 告警 / 上报等）
    raise e



# 对于for i, item in enumerate(sa.run(), start=1): 的解释：
# 现在run()方法会将每次vlm分析结果都透传出去，并且在结果字典里包含了：stream_segment_index字段，代表该流自任务启动以来的连续窗口序号; window_index_in_stream, 
# 当前轮中该流第几个窗口. 但是在透传出去的数据结构中并不包括当前RTSP任务一共运行了多少个窗口了, 因此由 for i, item in enumerate(sa.run(), start=1) 中的start
# 参数来计数; 本质上是为了模拟上层调用在合适触发update_polling_batch_interval等运行时控制方法, 上层完全可以定义其他复杂的触发逻辑, 比如与定时器联动, 与规则引擎联动。

# 就算没有提供计算RTSP安防总的切窗数量的字段和方法, 上层任然可以通过以下公式计算得到：
"""
【如何在上层计算当前 RTSP 安防任务的“总切窗数量”（跨所有轮次的累积窗口数）】

系统没有提供“跨全部流与全部轮次”的全局窗口编号字段，
但上层仍然可以通过以下公式自行计算得到：

    rtsp_total_cut_window_number
        = (每一轮轮询任务的切窗数量) * polling_round_index
          + (当前轮已切出的窗口数量)

其中三个量的来源：

1) 每一轮轮询任务的切窗数量（固定值）
   = sum(rtsp.rtsp_cut_number for rtsp in polling_list)

2) polling_round_index
   - 当前处于第几轮轮询（全局统一、从 0 开始）
   - 由系统在每个分析结果 item 中透传：item["polling_round_index"]

3) 当前轮已切出的窗口数量
   = 对当前到达的 item 按顺序数第几个窗口
   - 即：当前轮的“窗口计数”（轮内顺序）
   - 可由上层在循环中自行递增统计，或根据业务需求命名为：
         current_round_window_count
   - 不依赖系统字段、自行管理即可

举例：
    假设 polling_list 中有 3 路流，它们的 rtsp_cut_number 分别为：
        [1, 2, 1]
    则每一轮轮询总共会切：1 + 2 + 1 = 4 个窗口。

    如果当前 item 的 polling_round_index = 2（即正在跑第 3 轮）
    且当前轮已经产出了 3 个窗口（current_round_window_count = 3）

    则总窗口数量计算为：
        total = 4 * 2 + 3 = 11

    这表示：从任务启动到现在，总共切出了 11 个窗口。

该方法完全基于系统已提供的字段与轮询行为，上层无需修改底层逻辑即可获得全局统一的窗口计数。
"""









