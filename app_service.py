# -*- coding: utf-8 -*-
from __future__ import annotations

import os, json, uuid, time, threading, subprocess
from typing import Optional, Dict, Callable, Literal, List
from queue import Queue, Empty
from shutil import which
from pathlib import Path

from fastapi import FastAPI, Form, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel, Field

from streaming_analyze import StreamingAnalyze
from src.all_enum import (
    MODEL,
    VLM_DETECT_EVENT_LEVEL,
    LOCAL_VLM_MODEL_NAME,
    CLOUD_VLM_MODEL_NAME,
    VLM_SYSTEM_PROMPT_PRESET,
)
from src.configs.vlm_config import VlmConfig
from src.configs.cut_config import CutConfig
from src.configs.asr_config import AsrConfig
from src.utils.logger_utils import get_logger


from src.configs.rtsp_batch_config import RTSPBatchConfig, RTSP


logger = get_logger(__name__)

# ================= 基础 & 静态 =================
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
INDEX_HTML = os.path.join(BASE_DIR, "index.html")

app = FastAPI(title="JetLinksAI Video Inspection Service (Edge Box)", version="1.0-edge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

STATIC_DIR = os.path.join(BASE_DIR, "static")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

UPLOAD_DIR = os.path.join(BASE_DIR, "uploads") # 存放用户上传的离线文件
os.makedirs(UPLOAD_DIR, exist_ok=True)

VIDEO_EXTS = {".mp4", ".avi", ".mkv", ".mov", ".flv"}
AUDIO_EXTS = {".mp3", ".wav", ".aac", ".flac", ".ogg"}

def _save_upload(file: UploadFile) -> str:
    suffix = Path(file.filename or "").suffix or ""
    dst = os.path.join(UPLOAD_DIR, f"{uuid.uuid4().hex}{suffix}")
    with open(dst, "wb") as f:
        f.write(file.file.read())
    return os.path.abspath(dst)

def _ensure_local_media(file: Optional[UploadFile], file_path: Optional[str]) -> str:
    if file is not None and (file_path is not None):
        raise HTTPException(400, "file 与 file_path 只能二选一")
    if file is None and (file_path is None):
        raise HTTPException(400, "必须提供 file 或 file_path 之一")
    if file is not None:
        return _save_upload(file)
    p = os.path.abspath(file_path or "")
    if not os.path.exists(p):
        raise HTTPException(400, f"文件不存在: {p}")
    return p

# ================= SSE 简易 Broker =================
class SSEBroker:
    def __init__(self, max_queue_size: int = 1000):
        self._subs: Dict[str, Queue[str]] = {}
        self._filters: Dict[str, Optional[Callable[[dict], bool]]] = {}
        self._max = max_queue_size
        self._lock = threading.Lock()

    def subscribe(self, flt: Optional[Callable[[dict], bool]] = None) -> str:
        sid = uuid.uuid4().hex[:8]
        with self._lock:
            self._subs[sid] = Queue(self._max)
            self._filters[sid] = flt
        return sid

    def unsubscribe(self, sid: str):
        with self._lock:
            self._subs.pop(sid, None)
            self._filters.pop(sid, None)

    def publish(self, event: dict):
        line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
        with self._lock:
            for sid, q in self._subs.items():
                flt = self._filters.get(sid)
                if flt is not None:
                    try:
                        if not flt(event):
                            continue
                    except Exception:
                        continue
                try:
                    q.put_nowait(line)
                except Exception:
                    pass

    def stream(self, sid: str):
        q = self._subs.get(sid)
        if q is None:
            yield "event: end\ndata: {}\n\n"
            return
        try:
            yield f"event: hello\ndata: {json.dumps({'sid': sid})}\n\n"
            while True:
                try:
                    line = q.get(timeout=0.5)
                    yield line
                except Empty:
                    continue
        finally:
            self.unsubscribe(sid)

broker = SSEBroker()

# ================= Job 状态 =================
class Job:
    def __init__(self, ctrl: StreamingAnalyze, mode: MODEL):
        self.ctrl = ctrl
        self.mode = mode
        self.thread: Optional[threading.Thread] = None
        self.running = False
        self.viewer_proc: Optional[subprocess.Popen] = None
        self.start_ts = time.time()

JOBS: Dict[str, Job] = {}
JOBS_LOCK = threading.Lock()

def _get_job(jid: str) -> Job:
    with JOBS_LOCK:
        job = JOBS.get(jid)
    if not job:
        raise HTTPException(404, f"job_id 不存在：{jid}")
    return job

# —— 事件最小化：推视觉完成/ASR完成，并透传轮询元信息（若存在）
def _event_payload_min(job_id: str, ev: dict) -> Optional[dict]:
    typ = ev.get("type")
    seg = ev.get("segment_index")
    t0 = ev.get("clip_t0"); t1 = ev.get("clip_t1")
    ts = ((ev.get("_meta") or {}).get("emit_iso")) or time.strftime("%Y-%m-%dT%H:%M:%S")

    if typ == "vlm_stream_done":
        payload = {
            "type": "vlm_done",
            "job_id": job_id,
            "seg": seg,
            "text": (ev.get("full_text") or "").strip(),
            "t0": t0, "t1": t1, "ts": ts,
            "frame_pts": ev.get("frame_pts") or [],
            "evidence_images": ev.get("evidence_images") or [],
            "evidence_image_urls": ev.get("evidence_image_urls") or [],
        }
    elif typ == "asr_stream_done":
        payload = {
            "type": "asr_done",
            "job_id": job_id,
            "seg": seg,
            "text": (ev.get("full_text") or "").strip(),
            "t0": t0, "t1": t1, "ts": ts,
        }
    else:
        return None

    # 透传 A → B → 后端 → 前端 的轮询/分片元信息（存在才写入 payload）
    for k in (
        # --- 流级别元信息 ---
        "stream_index",          # 当前窗口所属的 RTSP 流在 polling_list 中的索引（0-based）
        "stream_url",            # 当前窗口所属的 RTSP 原始地址

        # --- 轮询级元信息 ---
        "polling_round_index",   # 当前是第几轮轮询（每跑完一轮，对所有流的窗口切片后 +1）
        "polling_batch_interval",# 两轮轮询之间的全局间隔秒数（polling 模式下可动态更新）

        # --- 分片级元信息 ---
        "stream_segment_index",  # 该流自任务启动以来的连续窗口序号（全局计数）
        "window_index_in_stream",# 该流在“当前轮”中是第几个窗口（局部计数）

        # --- 配置级元信息 ---
        "rtsp_cut_number",       # 该流在每一轮需要切的窗口数量（RTSP.rtsp_cut_number）
        "rtsp_system_prompt",    # 该流当前生效的提示词（预设/自定义/兜底从A侧获取）
    ):
        if k in ev:
            payload[k] = ev[k]
    return payload

def _should_push(ev_min: dict, job_mode: MODEL) -> bool:
    return ev_min.get("type") in ("vlm_done", "asr_done")

def _start_job(job_id: str, ctrl: StreamingAnalyze, mode: MODEL):
    job = Job(ctrl, mode)
    def _runner():
        job.running = True
        try:
            for ev in ctrl.run():
                ev_min = _event_payload_min(job_id, ev)
                if ev_min and _should_push(ev_min, job.mode):
                    broker.publish(ev_min)
        except Exception as e:
            broker.publish({"type": "job_error", "job_id": job_id, "msg": str(e)})
        finally:
            job.running = False
            broker.publish({"type": "job_end", "job_id": job_id})
    t = threading.Thread(target=_runner, name=f"job-{job_id}", daemon=True)
    job.thread = t
    with JOBS_LOCK:
        JOBS[job_id] = job
    t.start()
    return job_id

# ================= 构造配置 =================
def _build_vlm_config_for_security(
    *, vlm_backend: Literal["local","cloud"],
    local_model_name: Optional[str],
    cloud_model_name: Optional[str],
    vlm_event_min_level: str,
) -> VlmConfig:
    level = VLM_DETECT_EVENT_LEVEL[vlm_event_min_level.strip().upper()]
    if vlm_backend == "local":
        local_enum = LOCAL_VLM_MODEL_NAME[local_model_name.strip().upper()] if local_model_name else None
        if not local_enum: raise HTTPException(400, "local 模式需要提供 local_model_name")
        return VlmConfig(
            vlm_backend="local",
            vlm_local_model_name=local_enum,
            vlm_system_prompt=None,  # 每流自带
            vlm_streaming=False,
            vlm_task_history_enabled=False,
            vlm_event_min_level=level,
        )
    else:
        cloud_enum = CLOUD_VLM_MODEL_NAME[cloud_model_name.strip().upper()] if cloud_model_name else None
        if not cloud_enum: raise HTTPException(400, "cloud 模式需要提供 cloud_model_name")
        return VlmConfig(
            vlm_backend="cloud",
            vlm_cloud_model_name=cloud_enum,
            vlm_system_prompt=None,  # 每流自带
            vlm_streaming=False,
            vlm_task_history_enabled=False,
            vlm_event_min_level=level,
            )

def _build_cut_config(cut_window_sec: float, alpha_bgr: float, topk_frames: int) -> CutConfig:
    return CutConfig(cut_window_sec=float(cut_window_sec), alpha_bgr=float(alpha_bgr), topk_frames=int(topk_frames))

def _coerce_prompt(p: Optional[object]) -> Optional[object]:
    if isinstance(p, str):
        name = p.strip().upper()
        if name in VLM_SYSTEM_PROMPT_PRESET.__members__:
            return VLM_SYSTEM_PROMPT_PRESET[name]
        return p.strip()
    return p

# ================== 请求模型 ==================
class PollingStartIn(BaseModel):
    vlm_backend: Literal["local", "cloud"] = "local"
    local_model_name: Optional[str] = "qwen3-vl-8b-instruct"
    cloud_model_name: Optional[str] = None
    cut_window_sec: float = 4.0
    alpha_bgr: float = 0.5
    topk_frames: int = 1
    vlm_event_min_level: str = "LOW"
    polling_batch_interval: int = Field(15, ge=10, description="轮询批间隔(秒，≥10)")
    streams: List[RTSP] = Field(..., min_items=2, max_items=50)

# ================== SSE ==================
@app.get("/events")
def sse_events(job_id: Optional[str] = None):
    sid = broker.subscribe((lambda e: e.get("job_id") == job_id) if job_id else None)
    return StreamingResponse(
        broker.stream(sid), media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"},
    )

# ================== 单流启动 ==================
@app.post("/security/single_start")
async def security_single_start(
    rtsp_url: str = Form(...),
    vlm_backend: Literal["local", "cloud"] = Form("local"),
    local_model_name: Optional[str] = Form(None),
    cloud_model_name: Optional[str] = Form(None),
    # 两种方式：要么直接传每流自定义/预设名；要么传一个全局预设名
    rtsp_system_prompt: Optional[str] = Form(None),
    vlm_preset: Optional[str] = Form("SECURITY_DETECT_EVENTS"),
    cut_window_sec: float = Form(4.0),
    alpha_bgr: float = Form(0.5),
    topk_frames: int = Form(1),
    vlm_event_min_level: str = Form("LOW"),
    open_local_player: bool = Form(False),
    rtsp_cut_number: int = Form(1),
):
    if not (rtsp_url.startswith("rtsp://") or rtsp_url.startswith("rtsps://")):
        raise HTTPException(400, "rtsp_url 必须以 rtsp:// 或 rtsps:// 开头")

    vlm_cfg = _build_vlm_config_for_security(
        vlm_backend=vlm_backend,
        local_model_name=local_model_name,
        cloud_model_name=cloud_model_name,
        vlm_event_min_level=vlm_event_min_level,
    )
    cut_cfg = _build_cut_config(cut_window_sec, alpha_bgr, topk_frames)

    prompt = _coerce_prompt(rtsp_system_prompt) if rtsp_system_prompt else None
    if prompt is None and vlm_preset:
        preset_name = vlm_preset.strip().upper()
        if preset_name in VLM_SYSTEM_PROMPT_PRESET.__members__:
            prompt = VLM_SYSTEM_PROMPT_PRESET[preset_name]
        else:
            raise HTTPException(400, f"未知预设：{vlm_preset}")

    batch = RTSPBatchConfig(
        polling_list=[RTSP(rtsp_url=rtsp_url, rtsp_system_prompt=prompt, rtsp_cut_number=int(rtsp_cut_number))],
        polling_batch_interval=15,
        vlm_config=vlm_cfg,
    )

    ctrl = StreamingAnalyze(
        mode=MODEL.SECURITY_SINGLE,
        enable_b=True, enable_c=False,
        rtsp_batch_config=batch,
        cut_config=cut_cfg,
        vlm_config=vlm_cfg,
        asr_config=None,
    )

    job_id = uuid.uuid4().hex[:8]
    _start_job(job_id, ctrl, MODEL.SECURITY_SINGLE)

    viewer = "disabled"
    if open_local_player:
        proc = _launch_local_rtsp_viewer(rtsp_url, title=f"RTSP-{job_id}")
        if proc is not None:
            viewer = "ffplay" if which("ffplay") else ("vlc" if which("vlc") else "unknown")
        with JOBS_LOCK:
            job = JOBS.get(job_id)
            if job: job.viewer_proc = proc

    return {
        "ok": True, "job_id": job_id, "mode": "security-single",
        "rtsp": rtsp_url, "model": (local_model_name or cloud_model_name),
        "cut_window_sec": float(cut_window_sec), "topk_frames": int(topk_frames),
        "viewer": viewer,
    }

# ================== 轮询启动（streams 直接用你的 RTSP） ==================
@app.post("/security/polling_start")
async def security_polling_start(body: PollingStartIn):
    if len(body.streams) < 2 or len(body.streams) > 50:
        raise HTTPException(400, "轮询模式需要 2~50 条流")

    # 每路提示词：把字符串预设名映射到 Enum；自定义文本直接透传
    mapped_streams: List[RTSP] = []
    for s in body.streams:
        mapped_streams.append(
            RTSP(
                rtsp_url=s.rtsp_url,
                rtsp_system_prompt=_coerce_prompt(s.rtsp_system_prompt),
                rtsp_cut_number=s.rtsp_cut_number,
            )
        )

    vlm_cfg = _build_vlm_config_for_security(
        vlm_backend=body.vlm_backend,
        local_model_name=body.local_model_name,
        cloud_model_name=body.cloud_model_name,
        vlm_event_min_level=body.vlm_event_min_level,
    )
    cut_cfg = _build_cut_config(body.cut_window_sec, body.alpha_bgr, body.topk_frames)

    batch = RTSPBatchConfig(
        polling_list=mapped_streams,
        polling_batch_interval=int(body.polling_batch_interval),
        vlm_config=vlm_cfg,
    )

    ctrl = StreamingAnalyze(
        mode=MODEL.SECURITY_POLLING,
        enable_b=True, enable_c=False,
        rtsp_batch_config=batch,
        cut_config=cut_cfg,
        vlm_config=vlm_cfg,
        asr_config=None,
    )

    job_id = uuid.uuid4().hex[:8]
    _start_job(job_id, ctrl, MODEL.SECURITY_POLLING)
    return {
        "ok": True,
        "job_id": job_id,
        "mode": "security-polling",
        "streams": [s.rtsp_url for s in mapped_streams],
        "polling_batch_interval": int(body.polling_batch_interval),
        "cut_window_sec": float(body.cut_window_sec),
        "topk_frames": int(body.topk_frames),
    }

# ================== 运行期控制 & 动态增删改流 ==================
@app.post("/security/{job_id}/pause")
async def security_pause(job_id: str):
    job = _get_job(job_id)
    job.ctrl._broadcast_ctrl({"type": "PAUSE"})
    return {"ok": True, "job_id": job_id, "state": "paused"}

@app.post("/security/{job_id}/resume")
async def security_resume(job_id: str):
    job = _get_job(job_id)
    job.ctrl._broadcast_ctrl({"type": "RESUME"})
    return {"ok": True, "job_id": job_id, "state": "running"}

class UpdateStreamIn(BaseModel):
    old_rtsp_url: Optional[str] = None
    index: Optional[int] = Field(None, ge=0)
    item: RTSP

@app.post("/security/{job_id}/polling/add_stream")
async def polling_add_stream(job_id: str, item: RTSP):
    job = _get_job(job_id)
    if job.mode != MODEL.SECURITY_POLLING:
        raise HTTPException(400, "仅轮询模式支持动态新增流")

    cfg = job.ctrl.rtsp_batch_config
    urls = [getattr(x, "rtsp_url", None) for x in cfg.polling_list]
    if item.rtsp_url in urls:
        raise HTTPException(400, f"已存在重复的 rtsp_url：{item.rtsp_url}")
    if len(cfg.polling_list) >= 50:
        raise HTTPException(400, "SECURITY_POLLING 下最多 50 路")

    new_obj = RTSP(
        rtsp_url=item.rtsp_url,
        rtsp_system_prompt=_coerce_prompt(item.rtsp_system_prompt),
        rtsp_cut_number=item.rtsp_cut_number,
    )
    cfg.polling_list.append(new_obj)
    job.ctrl._send_rtsp_mode_message_to_a({"type": "RTSP_ADD_STREAM", "item": new_obj.model_dump()})
    return {"ok": True, "job_id": job_id, "count": len(cfg.polling_list)}

@app.post("/security/{job_id}/polling/remove_stream")
async def polling_remove_stream(job_id: str, rtsp_url: str = Form(...)):
    job = _get_job(job_id)
    if job.mode != MODEL.SECURITY_POLLING:
        raise HTTPException(400, "仅轮询模式支持动态删除流")

    cfg = job.ctrl.rtsp_batch_config
    before = len(cfg.polling_list)
    cfg.polling_list[:] = [it for it in cfg.polling_list if getattr(it, "rtsp_url", None) != rtsp_url]
    after = len(cfg.polling_list)
    if after == before:
        raise HTTPException(404, f"未找到 rtsp_url：{rtsp_url}")
    if after < 2:
        raise HTTPException(400, "SECURITY_POLLING 至少保留 2 路")

    job.ctrl._send_rtsp_mode_message_to_a({"type": "RTSP_REMOVE_STREAM", "rtsp_url": rtsp_url})
    return {"ok": True, "job_id": job_id, "count": after}

@app.post("/security/{job_id}/update_stream")
async def update_stream(job_id: str, body: UpdateStreamIn):
    job = _get_job(job_id)
    cfg = job.ctrl.rtsp_batch_config
    if not cfg or not cfg.polling_list:
        raise HTTPException(500, "rtsp_batch_config 未初始化或空列表")

    new_obj = RTSP(
        rtsp_url=body.item.rtsp_url,
        rtsp_system_prompt=_coerce_prompt(body.item.rtsp_system_prompt),
        rtsp_cut_number=body.item.rtsp_cut_number,
    )

    if job.mode == MODEL.SECURITY_SINGLE:
        old_url = getattr(cfg.polling_list[0], "rtsp_url", None)
        cfg.polling_list[0] = new_obj
        job.ctrl._send_rtsp_mode_message_to_a({
            "type":"RTSP_UPDATE_STREAM","old_rtsp_url": old_url,"item": new_obj.model_dump(),"index": 0
        })
        return {"ok": True, "job_id": job_id, "old": old_url, "new": new_obj.rtsp_url}

    if job.mode == MODEL.SECURITY_POLLING:
        repl_idx = None
        if body.index is not None and 0 <= body.index < len(cfg.polling_list):
            repl_idx = body.index
        elif body.old_rtsp_url:
            for i, it in enumerate(cfg.polling_list):
                if getattr(it, "rtsp_url", None) == body.old_rtsp_url:
                    repl_idx = i; break
        if repl_idx is None:
            raise HTTPException(404, "未找到要替换的流")
        for j, it in enumerate(cfg.polling_list):
            if j == repl_idx: continue
            if getattr(it, "rtsp_url", None) == new_obj.rtsp_url:
                raise HTTPException(400, f"新 URL 与其他路重复：{new_obj.rtsp_url}")

        old_url = getattr(cfg.polling_list[repl_idx], "rtsp_url", None)
        cfg.polling_list[repl_idx] = new_obj
        job.ctrl._send_rtsp_mode_message_to_a({
            "type":"RTSP_UPDATE_STREAM","old_rtsp_url": old_url,"item": new_obj.model_dump(),"index": repl_idx
        })
        return {"ok": True, "job_id": job_id, "index": repl_idx, "old": old_url, "new": new_obj.rtsp_url}

    raise HTTPException(400, f"当前模式不支持：{job.mode}")

@app.post("/security/{job_id}/polling/update_interval")
async def polling_update_interval(job_id: str, polling_batch_interval: int = Form(...)):
    if polling_batch_interval < 10:
        raise HTTPException(400, "polling_batch_interval 需 ≥ 10 秒")
    job = _get_job(job_id)
    if job.mode != MODEL.SECURITY_POLLING:
        raise HTTPException(400, "仅轮询模式支持修改轮询间隔")

    cfg = job.ctrl.rtsp_batch_config
    cfg.polling_batch_interval = int(polling_batch_interval)
    job.ctrl._send_rtsp_mode_message_to_a({
        "type":"RTSP_UPDATE_INTERVAL","polling_batch_interval": int(polling_batch_interval)
    })
    return {"ok": True, "job_id": job_id, "polling_batch_interval": int(polling_batch_interval)}

# ================== 离线：视频/音频（拒绝图片） ==================
@app.post("/offline/vision_describe_start")
async def offline_vision_describe_start(
    file: Optional[UploadFile] = File(default=None),
    file_path: Optional[str] = Form(default=None),

    vlm_backend: Literal["local", "cloud"] = Form("local"),
    local_model_name: Optional[str] = Form(None),
    cloud_model_name: Optional[str] = Form(None),

    system_prompt: Optional[str] = Form(None),

    cut_window_sec: float = Form(6.0),
    alpha_bgr: float = Form(0.5),
    topk_frames: int = Form(3),
):
    media_path = _ensure_local_media(file, file_path)
    ext = Path(media_path).suffix.lower()
    is_video = ext in VIDEO_EXTS
    is_audio = ext in AUDIO_EXTS
    if not (is_video or is_audio):
        raise HTTPException(400, f"仅支持视频{sorted(VIDEO_EXTS)}与音频{sorted(AUDIO_EXTS)}，收到 {ext}")

    vlm_cfg: Optional[VlmConfig] = None
    if is_video:
        sys_prompt_value = (system_prompt or "请对当前视频或关键帧进行描述。").strip()
        if vlm_backend == "local":
            local_enum = LOCAL_VLM_MODEL_NAME[local_model_name.strip().upper()] if local_model_name else None
            if not local_enum: raise HTTPException(400, "local 模式需要提供 local_model_name")
            vlm_cfg = VlmConfig(
                vlm_backend="local",
                vlm_local_model_name=local_enum,
                vlm_system_prompt=sys_prompt_value,
                vlm_streaming=False,
                vlm_task_history_enabled=False,
                vlm_event_min_level=VLM_DETECT_EVENT_LEVEL.LOW,
            )
        else:
            cloud_enum = CLOUD_VLM_MODEL_NAME[cloud_model_name.strip().upper()] if cloud_model_name else None
            if not cloud_enum: raise HTTPException(400, "cloud 模式需要提供 cloud_model_name")
            vlm_cfg = VlmConfig(
                vlm_backend="cloud",
                vlm_cloud_model_name=cloud_enum,
                vlm_system_prompt=sys_prompt_value,
                vlm_streaming=False,
                vlm_task_history_enabled=False,
                vlm_event_min_level=VLM_DETECT_EVENT_LEVEL.LOW,
            )

    cut_cfg = _build_cut_config(cut_window_sec, alpha_bgr, topk_frames)

    mode_for_offline = getattr(MODEL, "OFFLINE", MODEL.SECURITY_SINGLE)
    ctrl = StreamingAnalyze(
        url=media_path,
        mode=mode_for_offline,
        enable_b=is_video,
        enable_c=is_audio,
        cut_config=cut_cfg,
        vlm_config=vlm_cfg,   # 音频时 None
        asr_config=AsrConfig(),
    )

    job_id = uuid.uuid4().hex[:8]
    _start_job(job_id, ctrl, mode_for_offline)
    return {
        "ok": True,
        "job_id": job_id,
        "mode": "offline-audio" if is_audio else "offline-vision",
        "path": media_path,
        "cut_window_sec": float(cut_window_sec),
        "topk_frames": int(topk_frames),
        "subscribe": f"/events?job_id={job_id}",
    }

# ================== 作业管理 & 健康检查 ==================
@app.post("/job/{job_id}/stop")
async def stop_job(job_id: str):
    job = _get_job(job_id)
    job.ctrl.force_stop("manual stop")
    try:
        if job.viewer_proc: job.viewer_proc.terminate()
    except Exception:
        pass
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    return {"ok": True, "stopped": job_id}

@app.get("/jobs")
async def list_jobs():
    with JOBS_LOCK:
        out = [{"job_id": jid,
                "mode": str(job.mode.value if hasattr(job.mode,"value") else str(job.mode)),
                "running": job.running, "start_ts": job.start_ts}
               for jid, job in JOBS.items()]
    return {"ok": True, "jobs": out}

@app.get("/job/{job_id}/status")
async def job_status(job_id: str):
    job = _get_job(job_id)
    return {
        "ok": True, "job_id": job_id,
        "mode": str(job.mode.value if hasattr(job.mode,"value") else str(job.mode)),
        "running": job.running, "start_ts": job.start_ts,
    }

@app.get("/job/{job_id}/stats")
async def job_stats(job_id: str):
    job = _get_job(job_id)
    st = job.ctrl.snapshot_stats()
    return {"ok": True, "job_id": job_id, "stats": st}

@app.post("/job/{job_id}/stats/reset")
async def job_stats_reset(job_id: str):
    job = _get_job(job_id)
    job.ctrl.reset_stats()
    return {"ok": True, "job_id": job_id, "reset": True}

@app.post("/stop")
async def stop_all():
    failed = []
    with JOBS_LOCK: ids = list(JOBS.keys())
    for jid in ids:
        try:
            job = JOBS.get(jid)
            if not job: continue
            try: job.ctrl.force_stop("manual stop")
            except Exception as e: failed.append(f"{jid}:{e}")
            try:
                if job.viewer_proc: job.viewer_proc.terminate()
            except Exception: pass
        finally:
            with JOBS_LOCK:
                JOBS.pop(jid, None)
    if failed:
        return JSONResponse({"ok": False, "error": "; ".join(failed)}, status_code=500)
    return {"ok": True, "stopped": ids}

@app.get("/healthz")
def healthz():
    return {"ok": True, "service": "JetLinksAI Analyze Service", "version": "0.8.0-edge"}

@app.get("/", response_class=HTMLResponse)
def index():
    if os.path.exists(INDEX_HTML):
        return FileResponse(INDEX_HTML, media_type="text/html; charset=utf-8")
    return HTMLResponse("<h3>index.html 未找到。请把前端页放到与 app_service.py 同级目录。</h3>", status_code=200)

# ======== 预览工具（可选） ========
def _launch_local_rtsp_viewer(rtsp_url: str, title: str) -> Optional[subprocess.Popen]:
    if which("ffplay"):
        try:
            return subprocess.Popen(["ffplay","-rtsp_transport","tcp","-autoexit","-loglevel","error","-window_title",title,rtsp_url])
        except Exception:
            pass
    if which("vlc"):
        try:
            return subprocess.Popen(["vlc","--quiet","--play-and-exit",rtsp_url,"--video-title",title])
        except Exception:
            pass
    return None

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app_service:app", host="0.0.0.0", port=8000, reload=True)

