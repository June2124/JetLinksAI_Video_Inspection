# JetLinksAI Video Inspection

多路视频智能巡检与离线视频解析后端服务。  

支持 **RTSP 实时视频流** 与 **本地音视频文件**，结合 **本地 / 云端多模态大模型（VLM）** 和 **语音识别（ASR）**，实现安防事件检测、画面理解与语音转写，并通过 **SSE** 将结果实时推送给前端或上层平台。

---

## 1. 项目定位与运行模式

### 1.1 项目定位

本项目可以作为企业级「视频巡检能力层」，主要用途包括：

- 工厂 / 园区 / 楼宇的 **安防巡检**（闯入、斗殴、烟火、电箱敞开等异常事件检测）；
- 大批量历史视频的 **离线分析与摘要**；
- 需要在算力受限设备（如 Sophon TPU 盒子）上运行 **本地 VLM + 本地 ASR** 的场景。

系统通过 HTTP 接口统一管理多路流，负责：

- 从 RTSP 或本地文件中 **按时间窗口切片**；
- 抽取 **关键帧**；
- 调用 **VLM 分析图像**；
- 调用 **ASR 转写音频**；
- 生成结构化事件与文本结果；
- 持久化「证据帧」图片，并通过 **SSE 事件流** 实时推送。

### 1.2 工作模式（`MODEL` 枚举）

在 `src/all_enum.py` 中定义了三个核心模式：

- `OFFLINE`
  - 离线音视频解析。
  - 支持视频 + 纯音频文件，不支持单张图片。
- `SECURITY_SINGLE`
  - 单路 RTSP 安防巡检。
- `SECURITY_POLLING`
  - 多路 RTSP 轮询巡检（支持 2–50 路）。

---

## 2. 对外服务接口（`app_service.py`）

主入口为 `app_service.py`，使用 **FastAPI** 暴露 HTTP API 与 SSE 事件流。

### 2.1 SSE 事件流

- **接口**：`GET /events?job_id=...`
- **说明**：  
  通过内部 `broker` 订阅对应 `job_id` 的事件队列，将各个 Worker 推送的事件以 **SSE** 格式返回给前端或上层系统。
- **事件内容**（视场景而定）：
  - 切片开始 / 结束；
  - 关键帧文件路径；
  - VLM 分析结果（事件类型、等级、描述文本等）；
  - ASR 转写结果；
  - 任务状态与异常信息。

### 2.2 单路安防：`/security/single_start`

- **接口**：`POST /security/single_start`
- **参数形式**：`Form` 表单
- **核心参数**：
  - `rtsp_url`：RTSP 流地址。
  - `vlm_backend`：`"local"` / `"cloud"`。
  - `local_model_name` / `cloud_model_name`：VLM 模型名。
  - `rtsp_system_prompt`：该路摄像头的系统提示词。
  - `vlm_preset`：可选场景预设名。
  - `cut_window_sec`：切片窗口时长（秒）。
  - `alpha_bgr`：背景融合参数。
  - `topk_frames`：每窗口选择的关键帧数。
  - `vlm_event_min_level`：最低上报事件等级。
  - `rtsp_cut_number`：每轮对该流切片的窗口数量。
  - `open_local_player`：是否在本机打开播放器（调试用）。
- **内部流程**：
  1. 组装单条 `RTSP` 配置并创建 `RTSPBatchConfig`。
  2. 创建 `CutConfig`、`VlmConfig` 等配置对象。
  3. 实例化 `StreamingAnalyze`，模式为 `MODEL.SECURITY_SINGLE`。
  4. 启动独立线程运行该任务。

### 2.3 多路轮询安防：`/security/polling_start`

- **接口**：`POST /security/polling_start`
- **参数形式**：JSON 请求体，对应 Pydantic 模型 `PollingStartIn`。
- **全局字段**：
  - `vlm_backend`：`"local"` / `"cloud"`。
  - `local_model_name` / `cloud_model_name`：VLM 模型。
  - `cut_window_sec`、`alpha_bgr`、`topk_frames`：切片相关参数。
  - `vlm_event_min_level`：最低事件等级。
  - `polling_batch_interval`：轮询批次间隔（秒，默认 15，要求 ≥ 10）。
- **流级字段**（数组 `streams: List[RTSP]`，长度 2–50）：
  - `rtsp_url`：流地址。
  - `rtsp_system_prompt`：支持传入预设名或自定义提示词。
  - `rtsp_cut_number`：该流每轮切片窗口数量。
- **内部流程**：
  1. 构造包含多路流的 `RTSPBatchConfig`。
  2. 初始化对应配置与 `StreamingAnalyze`，模式为 `MODEL.SECURITY_POLLING`。
  3. 启动工作线程，并将 `job_id` 返回给调用方。

### 2.4 轮询任务管理接口

针对 `SECURITY_POLLING` 模式提供一组运行时管理接口：

- **暂停 / 恢复任务**
  - `POST /security/{job_id}/pause`
  - `POST /security/{job_id}/resume`
  - 通过内部控制队列通知 Worker，实现任务级别暂停 / 恢复。

- **动态增删流**
  - `POST /security/{job_id}/polling/add_stream`
  - `POST /security/{job_id}/polling/remove_stream`
  - 支持在任务运行期间动态添加 / 删除 RTSP 流：
    - 添加时会检查 `rtsp_url` 唯一性及总路数（≤ 50）。
    - 删除后对应流将不再参与后续轮询。

- **更新单路流参数**
  - `POST /security/{job_id}/update_stream`
  - 可调整某条流的 `rtsp_cut_number`、提示词等。

- **更新轮询间隔**
  - `POST /security/{job_id}/polling/update_interval`
  - 修改 `polling_batch_interval`，并通过控制队列同步给 Worker A。

### 2.5 离线音视频解析：`/offline/vision_describe_start`

- **接口**：`POST /offline/vision_describe_start`
- **输入方式**：
  - 上传文件 `file`；或
  - 传入服务器本地路径 `file_path`。
- **配置字段**：
  - `vlm_backend`、`local_model_name`、`cloud_model_name`。
  - 切片参数：`cut_window_sec`、`alpha_bgr`、`topk_frames`。
  - 可选 ASR / VLM 相关参数。
- **模式**：`MODEL.OFFLINE`
- **行为**：
  - 视频文件：同时走 VLM + ASR；
  - 纯音频文件：仅走 ASR；
  - 单张图片：明确拒绝。
- **输出**：
  - SSE 中实时返回各切片的画面描述 / 事件，以及对应音频转写；
  - 本地生成视频切片与关键帧文件。

### 2.6 Job 管理与健康检查

- **停止任务**：`POST /job/{job_id}/stop`
  - 停止三类 Worker 线程；
  - 释放内部资源；
  - 关闭本地播放器（如使用）。

- **查看任务列表**：`GET /jobs`
  - 返回当前所有任务的 `job_id`、模式、运行状态、启动时间等。

- **任务状态**：`GET /job/{job_id}/status`
- **统计信息**：
  - `GET /job/{job_id}/stats`
  - `POST /job/{job_id}/stats/reset`
  - 统计项包括调用次数、平均时延等。

- **全局停止**：`POST /stop`  
- **健康检查**：`GET /healthz`  
- **根路径**：`GET /`
  - 返回 `index.html` 示例页面，可用于简单调试。

---

## 3. 核心流水线与 Worker 设计

整体处理流程由 `streaming_analyze.py` 中的 `StreamingAnalyze` 协调，内部采用 **多队列 + 多 Worker** 的流水线结构。

### 3.1 `StreamingAnalyze` 职责

`StreamingAnalyze` 是整个系统的「调度中心」，主要职责包括：

- 在初始化时：
  - 注入运行模式（`OFFLINE` / `SECURITY_SINGLE` / `SECURITY_POLLING`）；
  - 注入 `RTSPBatchConfig`、`CutConfig`、`VlmConfig`、`AsrConfig`；
  - 注入 `runtime_machine_config`（包括 `LocalVlmRuntimeMachine` 等）。
- 创建内部队列，包括：
  - `Q_VIDEO`：A → B（视频切片 & 关键帧信息）；
  - `Q_AUDIO`：A → C（音频切片）；
  - `Q_VLM`：B 输出；
  - `Q_ASR`：C 输出；
  - `Q_CTRL_A/B/C`：控制队列（暂停、停止、更新 RTSP、调整间隔等）；
  - `Q_EVENTS`：统一事件总线，供 SSE 消费。
- 使用特殊的 `_STOP` 对象作为各 Worker 的退出信号。
- 在 `run()` / `run_stream()` 中：
  - 启动 Worker A/B/C；
  - 循环消费 `Q_EVENTS`，并：
    - 推送给 SSE；
    - 汇总统计信息；
    - 监控异常并进行清理。

### 3.2 Worker A：视频切片 & 关键帧提取（`worker_a_cut.py`）

**数据源**：

- RTSP 模式：
  - 从 `RTSPBatchConfig.polling_list` 中按轮询策略拉流；
  - 支持多路轮询，受 `polling_batch_interval` 约束。
- OFFLINE 模式：
  - 从本地视频文件读取。

**核心功能**：

- 使用 FFmpeg / OpenCV 将视频按 `cut_window_sec` 切成连续片段。
- 对每个片段：
  - 选取 `topk_frames` 张关键帧；
  - 可按 `alpha_bgr` 对背景进行融合；
  - 将片段信息写入 `Q_VIDEO`、音频信息写入 `Q_AUDIO`；
  - 同时将关键事件写入 `Q_EVENTS`（如切片开始 / 结束、关键帧路径）。

**控制能力**：

- 监听 `Q_CTRL_A`，响应：
  - 任务级暂停 / 恢复；
  - STOP 信号；
  - RTSP 轮询模式下的增删流、更新 `polling_batch_interval` 等指令。

### 3.3 Worker B：VLM 分析（`worker_b_vlm.py`）

**输入**：

- 从 `Q_VIDEO` 读取切片与关键帧元信息。

**处理逻辑**：

- 根据 `VlmConfig` 决定调用：
  - `CloudVLMClient`：云端大模型（DashScope 等）；
  - `LocalVLMClient`：本地部署的 Qwen3-VL 等模型（Sophon TPU）。
- 系统提示词策略：
  - OFFLINE 模式：使用 `offline_system_prompt`；
  - 安防模式：
    - 每路 RTSP 使用对应的 `rtsp_system_prompt`；
    - 或使用 `VLM_SYSTEM_PROMPT_PRESET` 中的预设场景提示词。

**输出**：

- 安防检测场景：
  - 输出结构化事件结果：
    - `type`：事件类型（闯禁区、烟雾/明火、电箱敞开等）；
    - `level`：事件等级；
    - 对应帧索引 / 图片 URL 等。
  - 根据 `vlm_event_min_level` 进行过滤，低于阈值的事件不推送。
- 一般描述场景：
  - 输出多段文字描述；
  - 使用文本归一化和去重逻辑，避免重复内容。

**证据帧导出**：

- 将重要关键帧复制到：
  - 目录：`static/evidence_images/...`
  - URL 前缀由 `VlmConfig.vlm_static_evidence_images_url_prefix` 控制（默认 `/static/evidence_images`）。
- 将事件及证据帧信息写入 `Q_EVENTS`，供前端展示与追溯。

### 3.4 Worker C：ASR 转写（`worker_c_asr.py`）

**输入**：

- 从 `Q_AUDIO` 消费音频切片。

**处理逻辑**：

- 根据 `AsrConfig.asr_backend` 选择：
  - 云端 ASR：`CLOUD_ASR_MODEL_NAME`；
  - 本地 ASR：`LOCAL_ASR_MODEL_NAME`。
- 支持：
  - 按片段或句子维度的时间戳；
  - 结果排序与简单规范化等。

**输出**：

- 将转写结果写入 `Q_ASR`；
- 同步推送到 `Q_EVENTS`，供 SSE 使用。

---

## 4. 配置体系（`src/configs` 与 `src/all_enum`）

### 4.1 RTSP 相关配置（`rtsp_batch_config.py`）

#### `RTSP`

描述单路 RTSP 流，包括：

- `rtsp_url`：流地址；
- `rtsp_system_prompt`：该流绑定的系统提示词；
- `rtsp_cut_number`：每轮轮询该流的切片窗口数量。

#### `RTSPBatchConfig`

- `polling_list: List[RTSP]`：多路 RTSP 配置列表；
- `cut_window_sec`、`alpha_bgr`、`topk_frames`：切片配置（与 `CutConfig` 对应）；
- `polling_batch_interval`：轮询批次间隔（秒）；
- `vlm_config: VlmConfig`：全局 VLM 配置。

**校验逻辑**（Pydantic `model_validator`）：

- 安防模式下禁止开启 `vlm_streaming`；
- 若设置了 `offline_system_prompt` 会给出 warning，提示在 RTSP 模式中应使用 `rtsp_system_prompt`。

### 4.2 切片配置（`cut_config.py`）

`CutConfig` 主要字段：

- `cut_window_sec`：切片窗口时长；
- `alpha_bgr`：背景融合系数；
- `topk_frames`：每窗口保留的关键帧数量。

### 4.3 VLM 配置（`vlm_config.py`）

关键字段：

- `offline_system_prompt`：离线解析模式下的系统提示词（仅 `OFFLINE` 模式生效，长度限制 ≤ 300 字）。
- `vlm_cloud_model_name` / `vlm_local_model_name`：云端 / 本地 VLM 模型名。
- `vlm_backend: Literal["cloud", "local"]`：VLM 后端选择。
- `vlm_streaming: bool`：流式输出开关（当前在安防模式下禁止）。
- `vlm_task_history_enabled`：是否保留上下文历史。
- `vlm_event_min_level: VLM_DETECT_EVENT_LEVEL`：事件等级过滤阈值。
- `vlm_static_evidence_images_dir`：证据帧保存目录。
- `vlm_static_evidence_images_url_prefix`：证据帧 URL 前缀（对外访问路径）。

### 4.4 ASR 配置（`asr_config.py`）

主要字段：

- `asr_backend: Literal["cloud", "local"]`：选择云端或本地 ASR。
- 本地模式配置：
  - 本地 ASR 模型名等基础配置。
- 云端模式配置：
  - `disfluency_removal_enabled`：语气词过滤；
  - `semantic_punctuation_enabled`：语义断句；
  - 最大静音时间、标点预测、反归一化等。

**重要校验逻辑**：

- 当 `asr_backend="local"` 时：
  - 不允许设置云端专有字段（语义断句、最大静音时间等），否则抛出包含字段列表的清晰错误；
  - 以避免误配置导致行为不符合预期。

### 4.5 枚举定义（`all_enum.py`）

- `MODEL`：`OFFLINE` / `SECURITY_SINGLE` / `SECURITY_POLLING`。
- `SOURCE_KIND`：RTSP / FILE 等来源类型。
- `CLOUD_VLM_MODEL_NAME` / `LOCAL_VLM_MODEL_NAME`。
- `CLOUD_ASR_MODEL_NAME` / `LOCAL_ASR_MODEL_NAME`。
- `VLM_DETECT_EVENT_LEVEL`：事件等级（低 / 中 / 高等）。
- `VLM_SYSTEM_PROMPT_PRESET`：
  - 内置十个场景（如工厂安防等）；
  - 每个场景包含 5–6 条「监督目标」，统一规范检测任务与结果格式。

---

## 5. Runtime Machine（运行时调度器）

### 5.1 `LocalVlmRuntimeMachine`

**目标**：  
基于 **本地 VLM 实际推理延迟** 自动调节负载，防止 TPU / GPU 过载。

**输入**：

- 运行模式（尤其是 `SECURITY_POLLING`）；
- `RTSPBatchConfig`（用于估算每小时片段数）；
- 一组阈值与窗口参数：
  - `latency_window_size`
  - `auto_latency_thresholds`（是否自动校准）；
  - 或手工设置的 `low_latency_ms` / `high_latency_ms` / `panic_latency_ms`。

**核心逻辑**：

- 在滑动时间窗口内统计 VLM 推理时延；
- 根据平均延迟做自适应调节：
  - **延迟过高**：
    - 增大 `cut_window_sec`（减少每小时切片数）；
    - 轮询模式下增大 `polling_batch_interval`。
  - **延迟偏低**：
    - 减小 `cut_window_sec`；
    - 轮询模式下缩短 `polling_batch_interval`。
- 初始阶段根据「每小时 VLM 片段数」估算需要多少样本用于阈值校准。

### 5.2 `QueueRuntimeMachine`

- 目前为占位实现（`__init__` 基本为空）。
- 预留用于通过 **队列长度 / 排队时间** 驱动负载调度的扩展能力：
  - 根据队列积压情况进一步智能调节切片频率、轮询间隔、不同 job 优先级等。

---

## 6. 工程结构与辅助模块

### 6.1 目录结构概览

```bash
JetLinksAI-Video-Inspection/
├── .env                     # 环境变量示例
├── start.sh                 # 启动脚本（封装 uvicorn）
├── run/                     # 运行时 PID 等信息
├── logs/                    # 日志目录
├── static/
│   └── out/                 # 示例输出：视频切片与关键帧
├── out/
│   └── last_vlm_raw.txt     # 最近一次 VLM 原始结果样本
└── JetLinksAI/
    ├── app_service.py       # FastAPI 主入口 & HTTP/SSE 接口
    ├── streaming_analyze.py # 核心调度器（A/B/C Worker + 事件总线）
    ├── index.html           # 简单前端调试页面
    ├── requirements_business.txt
    ├── src/
    │   ├── all_enum.py      # 模式枚举、模型枚举、场景预设提示词
    │   ├── configs/         # 配置对象定义
    │   │   ├── rtsp_batch_config.py
    │   │   ├── vlm_config.py
    │   │   ├── asr_config.py
    │   │   ├── cut_config.py
    │   │   └── runtime_machine_config.py
    │   ├── workers/         # A/B/C Worker
    │   │   ├── worker_a_cut.py
    │   │   ├── worker_b_vlm.py
    │   │   └── worker_c_asr.py
    │   ├── runtime_machine/
    │   │   ├── local_vlm_runtime_machine.py
    │   │   └── queue_runtime_machine.py
    │   └── utils/
    │       ├── ffmpeg/      # FFmpeg 封装（含 Sophon 适配）
    │       ├── opencv/      # OpenCV 封装（含 Sophon 适配）
    │       ├── vlm_client/  # 本地 / 云端 VLM Client
    │       ├── asr_client/  # 本地 / 云端 ASR Client
    │       ├── file_cleanup.py
    │       └── logger_utils.py
    └── uploads/             # 离线上传文件存储目录
