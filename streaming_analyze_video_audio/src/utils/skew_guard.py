'''
Author: 13594053100@163.com
Date: 2025-10-09 10:00:57
LastEditTime: 2025-10-14 04:01:12
'''

# -*- coding: utf-8 -*-


from __future__ import annotations
import os
import time
from threading import RLock
from typing import Optional, Dict, Any

from src.utils.logger_utils import get_logger

logger = get_logger(__name__)


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


class TranscriptPlaybackSkewController:
    # 经过反复测试，如果开启VLM和ASR对齐器TranscriptPlaybackSkewController，会导致许多的增量结果被吞，反而破坏了原始输出的词序，建议注释掉本类代码，或请自行开发
    # """
    # 仅节流 + 交错最小间隔，不依赖媒体时间戳的“可见时序控制器”。

    # 参数：
    #     max_visual_skew_s : float
    #         仅保留存储（方案 A 不强约束），为未来媒体时间对齐做准备。
    #     max_emit_rate_hz  : float
    #         发射最大频率；<=0 表示不做频率限制。
    #     cross_gap_ms      : float
    #         VLM 与 ASR 交错最小间隔（毫秒）；<=0 禁用。
    # """

    # def __init__(
    #     self,
    #     *,
    #     max_visual_skew_s: float = None,
    #     max_emit_rate_hz: float = None,
    #     cross_gap_ms: float = None,
    # ) -> None:
    #     # 允许通过环境变量覆盖默认值
    #     if max_visual_skew_s is None:
    #         max_visual_skew_s = _env_float("EMIT_MAX_SKEW_S", 3.0)
    #     if max_emit_rate_hz is None:
    #         max_emit_rate_hz = _env_float("EMIT_RATE_LIMIT_HZ", 8.0)
    #     if cross_gap_ms is None:
    #         cross_gap_ms = _env_float("EMIT_CROSS_GAP_MS", 60.0)

    #     self.max_visual_skew_s: float = float(max_visual_skew_s)
    #     self.max_emit_rate_hz: float = float(max_emit_rate_hz)
    #     self.cross_gap_ms: float = float(cross_gap_ms)

    #     self.min_emit_interval: float = (
    #         1.0 / self.max_emit_rate_hz if self.max_emit_rate_hz > 0 else 0.0
    #     )

    #     # 线程安全
    #     self._lock = RLock()

    #     # 最近一次发射（墙钟时间）
    #     self._last_vlm_emit_wall_ts: float = 0.0
    #     self._last_asr_emit_wall_ts: float = 0.0

    #     # 播放器“已播放到的媒体时间戳”（方案 A 不使用，但对外保留，为未来升级做准备）
    #     self._playback_media_ts: float = 0.0

    #     # 统计指标（累计值）
    #     self._ctr = {
    #         "vlm_allowed": 0,
    #         "vlm_block_rate": 0,
    #         "vlm_block_cross": 0,
    #         "asr_allowed": 0,
    #         "asr_block_rate": 0,
    #         "asr_block_cross": 0,
    #     }

    #     # 汇总日志
    #     self._last_report_ts = 0.0
    #     self._report_interval_s = float(os.getenv("SKEW_REPORT_INTERVAL_S", "10"))  # 每10s汇总一次（0=关闭）
    #     self._log_blocks = os.getenv("SKEW_LOG_BLOCKS", "0") == "1"  # 是否逐条打印被拦截原因（默认关闭）

    # # ---------- 基础工具 ----------

    # def _rate_limited(self, now_ts: float, last_emit_ts: float) -> bool:
    #     """是否因为速率限制需要拦截"""
    #     if self.min_emit_interval <= 0:
    #         return False
    #     return (now_ts - last_emit_ts) < self.min_emit_interval

    # def _cross_gap_block(self, now_ts: float, other_last_emit_ts: float) -> bool:
    #     """是否因为跨模态最小间隔需要拦截"""
    #     if self.cross_gap_ms <= 0:
    #         return False
    #     # 与另一路的发射时间太近 => 拦截，等待下一次轮询放行
    #     return (now_ts - other_last_emit_ts) * 1000.0 < self.cross_gap_ms

    # def _maybe_report(self, now_ts: float) -> None:
    #     """周期性输出一行汇总日志，避免刷屏"""
    #     if self._report_interval_s <= 0:
    #         return
    #     if now_ts - self._last_report_ts >= self._report_interval_s:
    #         self._last_report_ts = now_ts
    #         c = self._ctr
    #         logger.info(
    #             "[skew] report: "
    #             "vlm_allow=%d rate_blk=%d cross_blk=%d | "
    #             "asr_allow=%d rate_blk=%d cross_blk=%d | "
    #             "min_interval=%.3fs cross_gap_ms=%.1f playback_ts=%.3f",
    #             c["vlm_allowed"], c["vlm_block_rate"], c["vlm_block_cross"],
    #             c["asr_allowed"], c["asr_block_rate"], c["asr_block_cross"],
    #             self.min_emit_interval, self.cross_gap_ms, self._playback_media_ts
    #         )

    # # ---------- 播放器进度（预留） ----------

    # def update_playback_ts(self, media_ts_s: float) -> None:
    #     """
    #     预留接口：上游可喂播放器当前媒体时间，方案 A 不使用，仅存储。
    #     方案 B 可用此值与片段 media_ts 对比，严格控制“发射领先播放”的偏差。
    #     """
    #     with self._lock:
    #         if media_ts_s > self._playback_media_ts:
    #             self._playback_media_ts = media_ts_s

    # # ---------- VLM 控制 ----------

    # def allow_emit_vlm(self, now_wall_ts: Optional[float] = None, *, last_asr_ts: Optional[float] = None) -> bool:
    #     """
    #     返回是否允许当前时刻发射 VLM 文本（增量/收尾皆用）。
    #     规则：
    #       1) 速率限制：与上次 VLM 发射间隔 >= min_emit_interval
    #       2) 交错限制：与最近一次 ASR 发射间隔 >= cross_gap_ms（可选）
    #     """
    #     now = now_wall_ts or time.time()
    #     with self._lock:
    #         # 1) 速率限制
    #         if self._rate_limited(now, self._last_vlm_emit_wall_ts):
    #             self._ctr["vlm_block_rate"] += 1
    #             if self._log_blocks:
    #                 logger.debug(
    #                     "[skew] VLM blocked: rate-limit (dt=%.3f < %.3f)",
    #                     now - self._last_vlm_emit_wall_ts, self.min_emit_interval
    #                 )
    #             self._maybe_report(now)
    #             return False

    #         # 2) 交错限制（若主控传入了 last_asr_ts，就以它为准；否则用我们内部记录）
    #         other_ts = last_asr_ts if isinstance(last_asr_ts, (int, float)) else self._last_asr_emit_wall_ts
    #         if self._cross_gap_block(now, other_ts):
    #             self._ctr["vlm_block_cross"] += 1
    #             if self._log_blocks:
    #                 logger.debug(
    #                     "[skew] VLM blocked: cross-gap (dt=%.1fms < %.1fms)",
    #                     (now - other_ts) * 1000.0, self.cross_gap_ms
    #                 )
    #             self._maybe_report(now)
    #             return False

    #         # 放行并记录
    #         self._last_vlm_emit_wall_ts = now
    #         self._ctr["vlm_allowed"] += 1
    #         self._maybe_report(now)
    #         return True

    # # ---------- ASR 控制 ----------

    # def allow_emit_asr(self, now_wall_ts: Optional[float] = None, *, last_vlm_ts: Optional[float] = None) -> bool:
    #     """
    #     返回是否允许当前时刻发射 ASR 文本（增量/收尾皆用）。
    #     规则：
    #       1) 速率限制：与上次 ASR 发射间隔 >= min_emit_interval
    #       2) 交错限制：与最近一次 VLM 发射间隔 >= cross_gap_ms（可选）
    #     """
    #     now = now_wall_ts or time.time()
    #     with self._lock:
    #         # 1) 速率限制
    #         if self._rate_limited(now, self._last_asr_emit_wall_ts):
    #             self._ctr["asr_block_rate"] += 1
    #             if self._log_blocks:
    #                 logger.debug(
    #                     "[skew] ASR blocked: rate-limit (dt=%.3f < %.3f)",
    #                     now - self._last_asr_emit_wall_ts, self.min_emit_interval
    #                 )
    #             self._maybe_report(now)
    #             return False

    #         # 2) 交错限制（若主控传入了 last_vlm_ts，就以它为准；否则用我们内部记录）
    #         other_ts = last_vlm_ts if isinstance(last_vlm_ts, (int, float)) else self._last_vlm_emit_wall_ts
    #         if self._cross_gap_block(now, other_ts):
    #             self._ctr["asr_block_cross"] += 1
    #             if self._log_blocks:
    #                 logger.debug(
    #                     "[skew] ASR blocked: cross-gap (dt=%.1fms < %.1fms)",
    #                     (now - other_ts) * 1000.0, self.cross_gap_ms
    #                 )
    #             self._maybe_report(now)
    #             return False

    #         # 放行并记录
    #         self._last_asr_emit_wall_ts = now
    #         self._ctr["asr_allowed"] += 1
    #         self._maybe_report(now)
    #         return True

    # # ---------- 实用方法 ----------

    # def reset(self) -> None:
    #     """复位内部状态（便于单实例串行多任务时手动清空；一般不需要，主控重建实例或重置状态即可）"""
    #     with self._lock:
    #         self._last_vlm_emit_wall_ts = 0.0
    #         self._last_asr_emit_wall_ts = 0.0
    #         # 播放器进度保留或清零均可，这里选择保留
    #         # self._playback_media_ts = 0.0
    #         # 计数器不清零，作为累计数据；如需清零请手动调用 clear_counters()

    # def clear_counters(self) -> None:
    #     """清空累计计数（例如每次任务开始前调用，做任务内统计）"""
    #     with self._lock:
    #         for k in self._ctr:
    #             self._ctr[k] = 0
    #         self._last_report_ts = 0.0

    # def snapshot_stats(self) -> Dict[str, Any]:
    #     """导出当前统计与关键配置（方便主控监控线程或收尾日志打印）"""
    #     with self._lock:
    #         return {
    #             "counters": dict(self._ctr),
    #             "config": {
    #                 "min_emit_interval_s": self.min_emit_interval,
    #                 "cross_gap_ms": self.cross_gap_ms,
    #                 "max_visual_skew_s": self.max_visual_skew_s,
    #                 "report_interval_s": self._report_interval_s,
    #                 "log_blocks": self._log_blocks,
    #             },
    #             "runtime": {
    #                 "last_vlm_emit_wall_ts": self._last_vlm_emit_wall_ts,
    #                 "last_asr_emit_wall_ts": self._last_asr_emit_wall_ts,
    #                 "playback_media_ts": self._playback_media_ts,
    #             }
    #         }

    # # 可选：动态调整
    # def set_report_interval(self, seconds: float) -> None:
    #     with self._lock:
    #         self._report_interval_s = max(0.0, float(seconds))

    # def set_log_blocks(self, enabled: bool) -> None:
    #     with self._lock:
    #         self._log_blocks = bool(enabled)
    pass
