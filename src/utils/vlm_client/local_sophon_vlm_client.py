from __future__ import annotations

"""
本地 Qwen3-VL 客户端

特性：
- 使用 requests 同步 HTTP 请求
- 每次调用有固定超时时间 VLM_HTTP_MAX_WAIT（秒）
- 失败重试固定 1 次（共尝试 2 次）
- 不做任何图片缩放，原图直接传给 HTTP VLM
- infer() 返回: ("nonstream", None, (final_text, usage))
  其中 usage 增加结构化状态：
    {
      "backend": "local_http",
      "status": "ok" | "timeout" | "http_error" | "other_error" | "stopped",
      "attempts": <int>,          # 实际尝试次数
      "elapsed_sec": <float>,     # 整体耗时
      "error": "<msg>"            # 仅失败时存在
    }
"""

import os
import time
from typing import Dict, List, Optional, Tuple, Iterator, Any
from queue import Queue, Empty

import cv2
import requests
from requests.exceptions import Timeout as RequestsTimeout, RequestException

from src.utils import logger_utils

logger = logger_utils.get_logger(__name__)

# ==== 常量 ====
# HTTP 服务地址 & 请求超时时间
VLM_HTTP_BASE = os.getenv("LOCAL_VLM_HTTP_BASE", "http://127.0.0.1:8899")

# 单次 HTTP 请求最大等待时间（秒）
VLM_HTTP_MAX_WAIT = float(os.getenv("LOCAL_VLM_HTTP_MAX_WAIT", "120"))

# 失败重试次数（1 表示失败后再试 1 次，共 2 次机会）
VLM_HTTP_MAX_RETRY = 1


# ----------------------------------------------------------------------
# 工具：心跳
# ----------------------------------------------------------------------
def _poll_ctrl_heartbeat(q_ctrl: Queue, stop: object) -> bool:
    """心跳检测：收到 STOP / SHUTDOWN 就返回 True"""
    try:
        msg = q_ctrl.get_nowait()
    except Empty:
        return False
    if msg is stop:
        return True
    if isinstance(msg, dict) and msg.get("type") in ("STOP", "SHUTDOWN"):
        return True
    try:
        q_ctrl.put_nowait(msg)
    except Exception:
        pass
    return False


def _strip_file_scheme(p: str) -> str:
    """去掉 file:// 前缀"""
    if isinstance(p, str) and p.lower().startswith("file://"):
        return p[len("file://"):]
    return p


class LocalVLMClient:
    """
    同步版 Qwen3-VL HTTP 客户端
    """

    def __init__(
        self,
        model_name: str,
        messages: List[Dict[str, Any]],
        *,
        q_ctrl: Optional[Queue] = None,
        stop: Optional[object] = None,
    ):
        self.model_name = model_name
        self.messages = messages
        self.q_ctrl = q_ctrl
        self.stop = stop

        self._on_heartbeat = (
            (lambda: _poll_ctrl_heartbeat(q_ctrl, stop))
            if q_ctrl is not None and stop is not None
            else None
        )

    # ------------------------------------------------------------------
    # 消息转换：JetLinks 内部格式 -> OpenAI 风格 HTTP messages
    # ------------------------------------------------------------------
    @staticmethod
    def _build_http_messages_and_media(
        raw_msgs: List[Dict[str, Any]],
    ) -> Tuple[List[Dict[str, Any]], Optional[str]]:

        system_parts: List[str] = []
        user_text_pieces: List[Dict[str, Any]] = []
        first_image_path: Optional[str] = None

        for msg in raw_msgs:
            role = msg.get("role", "user")
            content = msg.get("content")

            # system
            if role == "system":
                if isinstance(content, list):
                    for p in content:
                        if isinstance(p, dict) and "text" in p:
                            system_parts.append(str(p["text"]))
                elif isinstance(content, str):
                    system_parts.append(content)
                continue

            # user
            if role == "user" and isinstance(content, list):
                for p in content:
                    if not isinstance(p, dict):
                        continue

                    # 文本块
                    if "text" in p and p["text"]:
                        user_text_pieces.append(
                            {"type": "text", "text": str(p["text"])}
                        )
                        continue

                    # 图片块（只取第一张）
                    if "image" in p and p["image"]:
                        if first_image_path is None:
                            local_path = _strip_file_scheme(str(p["image"]))
                            if os.path.exists(local_path):
                                first_image_path = os.path.abspath(local_path)
                            else:
                                logger.warning(
                                    "[LocalVLMClient] 图片不存在: %s", local_path
                                )
                        continue

                    # video 当前不支持，仅日志提示
                    if "video" in p and p["video"]:
                        logger.warning(
                            "[LocalVLMClient] 收到 video，HTTP 模型忽略该视频输入"
                        )
                        continue

        http_msgs: List[Dict[str, Any]] = []

        # system
        system_prompt = "\n".join([x.strip() for x in system_parts if x.strip()])
        if system_prompt:
            http_msgs.append({"role": "system", "content": system_prompt})

        # user
        user_content = list(user_text_pieces)

        if first_image_path:
            user_content.append(
                {"type": "image_url", "image_url": {"url": first_image_path}}
            )

        if user_content:
            http_msgs.append({"role": "user", "content": user_content})

        return http_msgs, first_image_path

    # ------------------------------------------------------------------
    # 同步 HTTP 调用（单次）
    # ------------------------------------------------------------------
    def _run_inference_http_once(self) -> Tuple[str, str]:
        """
        单次 HTTP 调用本地 VLM 服务
        返回 (final_text, media_mode)，media_mode in {"image","text"}
        """

        # 调用前做一次 STOP 心跳检测
        if self._on_heartbeat and self._on_heartbeat():
            raise RuntimeError("LocalVLMClient stopped before request")

        http_messages, first_image_path = self._build_http_messages_and_media(
            self.messages
        )

        if not http_messages:
            raise ValueError("LocalVLMClient: http_messages 为空")

        media_mode = "image" if first_image_path else "text"

        payload = {
            "model": self.model_name or "qwen3-vl-8b-instruct",
            "messages": http_messages,
            "stream": False,
        }

        url = f"{VLM_HTTP_BASE.rstrip('/')}/v1/chat/completions"
        logger.info("[LocalVLMClient] 调用本地 VLM: url=%s, mode=%s", url, media_mode)

        # timeout 使用 VLM_HTTP_MAX_WAIT，保证客户端最长等待时间
        resp = requests.post(url, json=payload, timeout=VLM_HTTP_MAX_WAIT)

        if resp.status_code != 200:
            try:
                err_json = resp.json()
            except Exception:
                err_json = resp.text
            raise RuntimeError(f"HTTP {resp.status_code}: {err_json}")

        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            raise ValueError("LocalVLMClient: 返回 choices 为空")

        msg = choices[0].get("message") or {}
        content = msg.get("content", "")

        if not isinstance(content, str):
            content = str(content)

        final_text = content.strip() or "抱歉，本地模型未生成有效回复。"

        return final_text, media_mode

    # ------------------------------------------------------------------
    # 对外 infer：带重试 + 日志 + 结构化 usage
    # ------------------------------------------------------------------
    def infer(
        self,
    ) -> Tuple[str, Optional[Iterator], Optional[Tuple[str, Optional[dict]]]]:
        """
        执行一次完整推理（同步版）
        返回: ("nonstream", None, (final_text, usage))
        """

        start = time.time()

        image_path = None
        video_path = None
        media_mode_for_log = None
        visual_tokens = 0

        # 仅用于日志：从原始 messages 中取第一张图片/视频路径
        for msg in self.messages:
            ct = msg.get("content")
            if isinstance(ct, list):
                for p in ct:
                    if "image" in p and p["image"]:
                        image_path = _strip_file_scheme(str(p["image"]))
                        media_mode_for_log = "image"
                        break
                    if "video" in p and p["video"]:
                        video_path = _strip_file_scheme(str(p["video"]))
                        media_mode_for_log = "video"
                        break
            if image_path or video_path:
                break

        # ====== 重试机制（同步） ======
        attempt = 0
        final_text = ""
        media_mode_http = "text"
        success = False
        last_error_kind: Optional[str] = None
        last_error_message: Optional[str] = None

        while attempt <= VLM_HTTP_MAX_RETRY:
            attempt += 1
            try:
                final_text, media_mode_http = self._run_inference_http_once()
                success = True
                last_error_kind = None
                last_error_message = None
                break  # 成功，跳出循环
            except Exception as e:
                # 区分错误类型：超时 / HTTP 请求错误 / 其他
                if isinstance(e, RequestsTimeout):
                    err_kind = "timeout"
                elif isinstance(e, RequestException):
                    err_kind = "http_error"
                else:
                    err_kind = "other_error"

                last_error_kind = err_kind
                last_error_message = str(e)

                logger.error(
                    "[LocalVLMClient] 推理失败 attempt=%d/%d, kind=%s, err=%s",
                    attempt,
                    VLM_HTTP_MAX_RETRY + 1,
                    err_kind,
                    e,
                )

                # 超时时给调用方一个统一前缀，方便主控侧做简单解析（不依赖 usage 也能兜底）
                if err_kind == "timeout":
                    final_text = f"[LOCAL_VLM_TIMEOUT] {e}"
                else:
                    final_text = f"[LOCAL_VLM_ERROR] {e}"

                if attempt > VLM_HTTP_MAX_RETRY:
                    logger.error(
                        "[LocalVLMClient] 达到最大重试次数，将本段标记为失败（%s），但仍返回错误状态给上游。",
                        err_kind,
                    )
                    break

        # ====== 日志统计：视觉 token 粗略估算 ======
        try:
            if media_mode_for_log == "image" and image_path and os.path.isfile(image_path):
                img = cv2.imread(image_path)
                if img is not None:
                    h, w = img.shape[:2]
                    visual_tokens = int(h * w / 32 / 32)
        except Exception as e:
            logger.warning("[LocalVLMClient] 视觉 token 估算失败: %s", e)

        elapsed = time.time() - start
        logger.info(
            "[LocalVLMClient] 模式=%s | 总耗时=%.3fs | vision_token≈%d | status=%s",
            media_mode_http,
            elapsed,
            visual_tokens,
            "ok" if success else (last_error_kind or "unknown"),
        )

        # 结构化 usage，供 B / 主控做“负载自适应”
        usage: dict = {
            "backend": "local_http",
            "status": "ok" if success else (last_error_kind or "unknown_error"),
            "attempts": attempt,
            "elapsed_sec": elapsed,
        }
        if not success and last_error_message:
            usage["error"] = last_error_message

        # 与旧接口兼容：mode="nonstream"，第二个返回值为 None
        return ("nonstream", None, (final_text, usage))
