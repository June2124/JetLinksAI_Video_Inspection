'''
Author: 13594053100@163.com
Date: 2025-10-13 18:55:39
LastEditTime: 2025-10-17 13:47:53
'''

from streaming_analyze import StreamingAnalyze
from src.all_enum import MODEL

# # 离线无声视频
# ctrl = StreamingAnalyze(
#     url=r"D:\streaming_analyze_video_audio\static\video\RAG_video_no_sound_test.mp4",
#     mode=MODEL.OFFLINE,
#     slice_sec=5,
#     enable_b=True,   # 只跑 VLM
#     enable_c=False,
#     skew_guard_enabled=None #关闭对齐与节流
# )

# 离线音频
# ctrl = StreamingAnalyze(
#     url=r"D:\streaming_analyze_video_audio\static\video\RAG_Video_with_sound_test_16k.wav",
#     mode=MODEL.OFFLINE,
#     slice_sec=5,
#     enable_b=False,  
#     enable_c=True,  # 只跑 ASR
#     skew_guard_enabled=None #关闭对齐与节流
# )

# 离线音视频
# ctrl = StreamingAnalyze(
#     url=r"D:\streaming_analyze_video_audio\static\video\RAG_Video_with_sound_test.mp4",
#     mode=MODEL.OFFLINE,
#     slice_sec=5,
#     enable_b=True,  
#     enable_c=True,
#     skew_guard_enabled=None #关闭对齐与节流 
# )

# 实时 RTSP MODEL.SECURITY安防模式示例（请把下面的示例地址换成你自己的真实 RTSP 地址）
ctrl = StreamingAnalyze(
    url="rtsp://admin:p@ssw0rd@192.168.33.152:554/Streaming/Channels/101",  # ← 请填写**实际RTSP地址**
    mode=MODEL.SECURITY,   # 实时流建议 ONLINE 或 SECURITY
    slice_sec=4,
    enable_b=True,       # 开 VLM
    enable_c=False,        # 有音轨且想做转写就开
    skew_guard_enabled=None #关闭对齐与节流
)

# 边跑边拿事件：VLM(增量+收尾)；ASR(仅收尾 full_text)
for ev in ctrl.run_stream(print_vlm=False, print_asr=False):
    print(ev)  # 关键字段：ev["type"], ev["segment_index"], ev.get("delta")/ev.get("full_text"), ev["_meta"]["emit_iso"]
    # 可按条件手动停止直播：
    # if need_stop: ctrl.force_stop("manual stop")

