# -*- coding: utf-8 -*-
from __future__ import annotations

import time
from collections import deque
from typing import Deque, Dict, Any, Callable, Optional, Tuple

from src.all_enum import MODEL
from src.configs.rtsp_batch_config import RTSPBatchConfig
from src.utils.logger_utils import get_logger

logger = get_logger(__name__)


class LocalVlmRuntimeMachine:
    """
    VLM 运行时状态机：
    - 上游只决定：是否启用状态机；
    - 内部根据 mode + rtsp_batch_config 自动选择：
        * 时间参数（latency_window_size / check_interval_events / min_adjust_gap_sec）、
        * 自适应延迟阈值（low/high/panic）、
        * 步长策略（小间隔/大间隔不同处理）；
    - 对外只暴露两个回调：
        * update_cut_window_sec(new_cut_sec)
        * update_polling_batch_interval(new_interval_sec)  # 仅轮询模式
    - 方案 B：当 avg_latency 连续 N 轮落在“正常区间”（low < avg < high）时，
      将 cut_window_sec / polling_batch_interval 直接 snap 回基础设置。
    """

    def __init__(
        self,
        mode: MODEL,
        *, 
        base_cut_window_sec: float,
        rtsp_batch_config: Optional[RTSPBatchConfig],
        update_cut_window_sec: Callable[[float], None],
        update_polling_batch_interval: Optional[Callable[[float], None]] = None,
        # ====== 下方参数视为“库内默认配置”，上游正常不传 ======
        # 统计窗口 / 检查频率 / 防抖时间下限（只是默认值，会在内部根据轮询间隔自动微调）
        latency_window_size: int = 30,
        check_interval_events: int = 10,
        min_adjust_gap_sec: float = 20.0,
        # 切窗大小可调范围（硬上下限）
        cut_window_range: Tuple[float, float] = (1.0, 60.0),
        # polling_interval 的“小范围阈值区间”（用于区分“小间隔场景”和“大间隔场景”）
        # 默认含义：
        #   (10, 3600) → interval ∈ [10s, 3600s] 视为“正常/中小间隔”，
        #                interval > 3600s 视为“大间隔”，步子会收小，并用%调整。
        min_polling_interval_range: Tuple[float, float] = (10.0, 3600.0),
        # 延迟阈值（None 表示交给状态机自适应；全部给定则视为“手动指定”，不再自动校准）
        low_latency_ms: Optional[float] = None,
        high_latency_ms: Optional[float] = None,
        panic_latency_ms: Optional[float] = None,
    ) -> None:
        if not isinstance(mode, MODEL):
            raise TypeError(f"mode 必须是 MODEL 枚举，实际为 {type(mode)}")

        if base_cut_window_sec <= 0:
            raise ValueError("base_cut_window_sec 必须为正数")

        # --- 模式与轮询配置 ---
        self.mode = mode
        self._rtsp_batch_config = rtsp_batch_config

        # SECURITY_POLLING 模式下一定要有有效的 polling_interval 和更新回调
        if self.mode == MODEL.SECURITY_POLLING:
            if rtsp_batch_config is None:
                raise ValueError("SECURITY_POLLING 模式下必须提供 rtsp_batch_config")
            if update_polling_batch_interval is None:
                raise ValueError("SECURITY_POLLING 模式下必须提供 update_polling_batch_interval 回调")
            base_polling_interval_sec = float(rtsp_batch_config.polling_batch_interval)
            if base_polling_interval_sec <= 0:
                raise ValueError("rtsp_batch_config.polling_batch_interval 必须为 >0 的秒数")
        else:
            # OFFLINE / SECURITY_SINGLE 默认认为无“轮询周期”概念
            base_polling_interval_sec = 0.0

        # --- 切窗 ---
        self.base_cut_window_sec = float(base_cut_window_sec)
        self.current_cut_window_sec = float(base_cut_window_sec)
        self.cut_window_min, self.cut_window_max = cut_window_range

        # --- 轮询间隔（仅轮询模式实际使用） ---
        self.base_polling_interval_sec = float(base_polling_interval_sec)
        self.current_polling_interval_sec = float(base_polling_interval_sec)
        self._small_interval_lo, self._small_interval_hi = min_polling_interval_range
        # 硬最小/最大间隔：保证不会缩得太离谱，也不会放得太夸张
        self._hard_min_polling_interval = max(5.0, self._small_interval_lo)
        hard_max_guess = max(
            self._small_interval_hi * 2.0,
            self.base_polling_interval_sec * 2.0,
            3600.0,  # 至少给到 1h
        )
        self._hard_max_polling_interval = min(hard_max_guess, 6 * 3600.0)  # 最多 6h

        # 回调
        self.update_cut_window_sec = update_cut_window_sec
        self.update_polling_batch_interval = update_polling_batch_interval

        # --- 延迟阈值：是否让状态机自适应 ---
        self._auto_latency_thresholds: bool = not (
            low_latency_ms is not None
            and high_latency_ms is not None
            and panic_latency_ms is not None
        )

        if self._auto_latency_thresholds:
            # 先占位，等采集到足够样本后再真正赋值
            self.low_latency_ms = 0.0
            self.high_latency_ms = float("inf")
            self.panic_latency_ms = float("inf")
            self._latency_threshold_calibrated = False
        else:
            self.low_latency_ms = float(low_latency_ms)  # type: ignore[arg-type]
            self.high_latency_ms = float(high_latency_ms)  # type: ignore[arg-type]
            self.panic_latency_ms = float(panic_latency_ms)  # type: ignore[arg-type]
            self._latency_threshold_calibrated = True

        # === 核心：根据 mode + base_polling_interval_sec 自动修正 timing 参数 ===
        (
            self.latency_window_size,
            self.check_interval_events,
            self.min_adjust_gap_sec,
        ) = self._auto_init_timing_params(
            mode=self.mode,
            base_polling_interval_sec=self.base_polling_interval_sec,
            default_latency_window_size=latency_window_size,
            default_check_interval_events=check_interval_events,
            default_min_adjust_gap_sec=min_adjust_gap_sec,
        )

        # === 基于 RTSPBatchConfig 估算“基线样本数” ===
        self._calibration_samples_target, est_segments_per_hour = self._compute_calibration_samples(
            mode=self.mode,
            rtsp_batch_config=self._rtsp_batch_config,
            base_cut_window_sec=self.base_cut_window_sec,
            latency_window_size=self.latency_window_size,
        )

        if self._auto_latency_thresholds:
            if est_segments_per_hour is not None:
                logger.info(
                    "[VLM-RUNTIME] 延迟阈值自适应：估算每小时 VLM 片段数≈%.1f，"
                    "计划使用前 %d 条 latency_ms 作为基线样本。",
                    est_segments_per_hour,
                    self._calibration_samples_target,
                )
            else:
                logger.info(
                    "[VLM-RUNTIME] 延迟阈值自适应：缺少有效 RTSP 配置，使用默认基线样本数=%d。",
                    self._calibration_samples_target,
                )
        else:
            logger.info(
                "[VLM-RUNTIME] 延迟阈值使用上游显式配置：low=%.0fms, high=%.0fms, panic=%.0fms",
                self.low_latency_ms,
                self.high_latency_ms,
                self.panic_latency_ms,
            )

        # 运行期状态
        self._latency_hist: Deque[float] = deque(maxlen=self.latency_window_size)
        self._event_counter: int = 0
        self._last_adjust_ts: float = 0.0

        # 方案 B：稳定区间计数
        self._stable_counter: int = 0
        # 连续多少“检查轮次”都处在正常区间，就 snap 回 base
        self._stable_rounds_to_snap: int = 3

        logger.info(
            "[VLM-RUNTIME] 初始化: mode=%s, base_cut=%.1fs, base_polling=%.1fs, "
            "latency_window_size=%d, check_interval_events=%d, min_adjust_gap_sec=%.1fs, "
            "polling_interval_hard_range=[%.1f, %.1f], calibration_samples_target=%d",
            self.mode.value,
            self.base_cut_window_sec,
            self.base_polling_interval_sec,
            self.latency_window_size,
            self.check_interval_events,
            self.min_adjust_gap_sec,
            self._hard_min_polling_interval,
            self._hard_max_polling_interval,
            self._calibration_samples_target,
        )

    # ------------------------------------------------------------------ #
    #         根据 mode + base_polling_interval_sec 自动定时参数         #
    # ------------------------------------------------------------------ #

    def _auto_init_timing_params(
        self,
        *,
        mode: MODEL,
        base_polling_interval_sec: float,
        default_latency_window_size: int,
        default_check_interval_events: int,
        default_min_adjust_gap_sec: float,
    ) -> Tuple[int, int, float]:
        """
        自动选择：
        - latency_window_size: 平均延迟采用多少个样本的窗口；
        - check_interval_events: 每多少个 vlm_done 事件评估一次；
        - min_adjust_gap_sec: 两次调参之间最小间隔（防抖）。
        """

        # OFFLINE / 单流场景：事件密集，直接用默认即可
        if mode in (MODEL.OFFLINE, MODEL.SECURITY_SINGLE) or base_polling_interval_sec <= 0:
            return (
                int(default_latency_window_size),
                int(default_check_interval_events),
                float(default_min_adjust_gap_sec),
            )

        interval = float(base_polling_interval_sec)

        # 高频轮询：<= 5min
        if interval <= 300.0:
            latency_ws = default_latency_window_size        # 30
            check_every = default_check_interval_events     # 10
            min_gap = max(default_min_adjust_gap_sec, 20.0)
            return int(latency_ws), int(check_every), float(min_gap)

        # 中频轮询：5min ~ 20min
        if interval <= 1200.0:
            latency_ws = max(12, default_latency_window_size // 2)   # ~15
            check_every = max(4, default_check_interval_events // 2) # ~5
            min_gap = max(default_min_adjust_gap_sec, 40.0)
            return int(latency_ws), int(check_every), float(min_gap)

        # 低频轮询：>20min，含“3 路 + 1 小时间隔”之类
        latency_ws = 12      # 类似覆盖最近 3~4 轮
        check_every = 6      # 大致 2 轮评估一次
        min_gap = max(default_min_adjust_gap_sec, 60.0)
        return int(latency_ws), int(check_every), float(min_gap)

    # ------------------------------------------------------------------ #
    #      根据 RTSPBatchConfig 估算“需要多少条样本做基线统计”           #
    # ------------------------------------------------------------------ #

    def _compute_calibration_samples(
        self,
        *,
        mode: MODEL,
        rtsp_batch_config: Optional[RTSPBatchConfig],
        base_cut_window_sec: float,
        latency_window_size: int,
    ) -> Tuple[int, Optional[float]]:
        """
        返回：
            (calibration_samples_target, est_segments_per_hour)

        - calibration_samples_target：阈值自适应阶段需要的样本条数；
        - est_segments_per_hour：估算的“每小时 VLM 片段数”（仅用于日志，可为 None）。
        """

        # 默认值：没有任何 RTSP 信息时，直接用窗口大小的一半
        default_target = max(10, latency_window_size // 2)

        if not rtsp_batch_config or not rtsp_batch_config.polling_list:
            return default_target, None

        # 1) 估算“每轮总切片数”
        total_cuts_per_round = 0
        try:
            for it in rtsp_batch_config.polling_list:
                cn = getattr(it, "rtsp_cut_number", 1) or 1
                total_cuts_per_round += int(cn)
        except Exception:
            return default_target, None

        total_cuts_per_round = max(2, total_cuts_per_round)

        # 2) 估算每小时产生多少个片段
        if mode == MODEL.SECURITY_POLLING:
            interval = float(rtsp_batch_config.polling_batch_interval or 0.0)
            if interval <= 0:
                return default_target, None
            rounds_per_hour = 3600.0 / interval
            est_segments_per_hour = total_cuts_per_round * rounds_per_hour
        else:
            # SECURITY_SINGLE：近似认为持续连续监控
            if base_cut_window_sec <= 0:
                return default_target, None
            est_segments_per_hour = total_cuts_per_round * (3600.0 / base_cut_window_sec)

        # 3) 根据“每小时片段数”的量级，决定需要多少样本
        sph = est_segments_per_hour

        if sph >= 80:
            # 负载很高，片段密集：用完整窗口 30 条做基线
            target = min(latency_window_size, 30)
        elif sph >= 30:
            # 中高负载：基线 20 条
            target = min(latency_window_size, 20)
        elif sph >= 10:
            # 中等负载：基线 12 条
            target = min(latency_window_size, 12)
        elif sph >= 4:
            # 低负载：基线 8 条
            target = min(latency_window_size, 8)
        else:
            # 非常低负载（例如 3 条/h，轮询间隔很大）：也至少要 6 条样本
            target = min(latency_window_size, 6)

        target = max(5, int(target))
        return target, est_segments_per_hour

    # ------------------------------------------------------------------ #
    #                           对外入口                                 #
    # ------------------------------------------------------------------ #

    def on_vlm_done(self, payload: Dict[str, Any]) -> None:
        """
        主控在收到一条 `vlm_stream_done` 后调用此方法。
        payload 需要包含：
            - latency_ms: float/int
        其他字段仅用于日志，不强制。
        """
        try:
            lat_ms = float(payload.get("latency_ms") or 0.0)
        except Exception:
            return
        if lat_ms <= 0:
            return

        self._latency_hist.append(lat_ms)
        self._event_counter += 1

        # ========== 阶段 1：延迟阈值自适应校准（只采样，不调参） ==========
        if self._auto_latency_thresholds and not self._latency_threshold_calibrated:
            self._try_calibrate_latency_thresholds()
            # 校准阶段不做任何 cut/polling 调整
            if not self._latency_threshold_calibrated:
                return
            # 刚刚完成校准这一条也不立即调，让下一条再进入正常逻辑
            return

        # ========== 阶段 2：正常基于平均延迟做调参 ==========
        # 样本太少不调
        if len(self._latency_hist) < max(5, self.check_interval_events // 2):
            return

        # 未到检查间隔也不调
        if (self._event_counter % self.check_interval_events) != 0:
            return

        now_ts = time.time()
        if now_ts - self._last_adjust_ts < self.min_adjust_gap_sec:
            return

        avg_lat = sum(self._latency_hist) / len(self._latency_hist)

        logger.info(
            "[VLM-RUNTIME] 检查触发: avg_latency=%.0fms (window=%d), "
            "cut=%.1fs, polling_interval=%.1fs, "
            "low=%.0f, high=%.0f, panic=%.0f",
            avg_lat, len(self._latency_hist),
            self.current_cut_window_sec,
            self.current_polling_interval_sec,
            self.low_latency_ms,
            self.high_latency_ms,
            self.panic_latency_ms,
        )

        # 高延迟 → 降负载（减轻模型压力）
        if avg_lat >= self.high_latency_ms:
            self._stable_counter = 0  # 离开稳定区间
            self._adjust_down_load(avg_latency_ms=avg_lat)
            self._last_adjust_ts = now_ts

        # 低延迟 → 加负载（提高时空覆盖率）
        elif avg_lat <= self.low_latency_ms:
            self._stable_counter = 0  # 离开稳定区间
            self._adjust_up_load(avg_latency_ms=avg_lat)
            self._last_adjust_ts = now_ts

        # 中间区间 → 认为处于“稳定状态”，累计次数，达到一定轮数后 snap 回基础设置
        else:
            self._stable_counter += 1
            logger.info(
                "[VLM-RUNTIME] 稳定区间计数：%d / %d（avg_latency=%.0fms 介于 low=%.0f 与 high=%.0f 之间）",
                self._stable_counter,
                self._stable_rounds_to_snap,
                avg_lat,
                self.low_latency_ms,
                self.high_latency_ms,
            )
            if self._stable_counter >= self._stable_rounds_to_snap:
                self._stable_counter = 0
                self._snap_back_to_base_if_needed(avg_latency_ms=avg_lat)
                self._last_adjust_ts = now_ts

    # ------------------------------------------------------------------ #
    #                 阶段 1：延迟阈值自适应（只执行一次）               #
    # ------------------------------------------------------------------ #

    def _try_calibrate_latency_thresholds(self) -> None:
        """
        在自适应模式下，根据前 N 条 latency_ms 计算：
            avg_latency_ms，
            low/high/panic = 0.7/1.5/3.0 * avg_latency_ms
        只在达到 _calibration_samples_target 时执行一次。
        """
        n = len(self._latency_hist)
        target = self._calibration_samples_target

        if n < max(3, target):
            # 采样进度日志（避免刷屏，只在若干关键点提示）
            if n in (1, max(2, target // 2)):
                logger.info(
                    "[VLM-RUNTIME] 延迟阈值自适应采样中：已收集 %d/%d 条 latency_ms 样本。",
                    n, target,
                )
            return

        avg_latency_ms = sum(self._latency_hist) / n
        base = max(avg_latency_ms, 1000.0)  # 防止因为偶发性极小值导致阈值过低

        low = base * 0.7
        high = base * 1.5
        panic = base * 3.0

        # 保险：确保单调递增
        if high <= low:
            high = low * 1.2
        if panic <= high:
            panic = high * 1.5

        self.low_latency_ms = low
        self.high_latency_ms = high
        self.panic_latency_ms = panic
        self._latency_threshold_calibrated = True

        logger.info(
            "[VLM-RUNTIME] 延迟阈值自适应完成：avg=%.0fms → "
            "low=%.0f, high=%.0f, panic=%.0f （样本数=%d）。",
            avg_latency_ms,
            self.low_latency_ms,
            self.high_latency_ms,
            self.panic_latency_ms,
            n,
        )

    # ------------------------------------------------------------------ #
    #                  方案 B：稳定多轮后 snap 回基础设置               #
    # ------------------------------------------------------------------ #

    def _snap_back_to_base_if_needed(self, *, avg_latency_ms: float) -> None:
        """
        当 avg_latency 连续多轮落在正常区间（low < avg < high）后调用：
        - 若当前 cut_window_sec 明显大于 base_cut_window_sec，则直接恢复为基础值；
        - 对 SEC_POLLING 模式，同理对 polling_batch_interval 做 snap。
        """
        changed = False

        # ---- cut_window_sec snap 回 base ----
        if self.current_cut_window_sec > self.base_cut_window_sec * 1.2:
            old = self.current_cut_window_sec
            self.current_cut_window_sec = self.base_cut_window_sec
            logger.info(
                "[VLM-RUNTIME] 稳定多轮后，将 cut_window_sec 从 %.1fs 直接恢复为基础值 %.1fs (avg_latency=%.0fms)",
                old,
                self.base_cut_window_sec,
                avg_latency_ms,
            )
            try:
                self.update_cut_window_sec(self.current_cut_window_sec)
            except Exception as e:
                logger.warning("[VLM-RUNTIME] snap 回调 update_cut_window_sec 异常: %s", e)
            changed = True

        # ---- polling_batch_interval snap 回 base（仅轮询模式） ----
        if (
            self.mode == MODEL.SECURITY_POLLING
            and self.update_polling_batch_interval is not None
            and self.base_polling_interval_sec > 0
            and self.current_polling_interval_sec > self.base_polling_interval_sec * 1.2
        ):
            old = self.current_polling_interval_sec
            self.current_polling_interval_sec = self.base_polling_interval_sec
            logger.info(
                "[VLM-RUNTIME] 稳定多轮后，将 polling_batch_interval 从 %.1fs 直接恢复为基础值 %.1fs (avg_latency=%.0fms)",
                old,
                self.base_polling_interval_sec,
                avg_latency_ms,
            )
            try:
                self.update_polling_batch_interval(self.current_polling_interval_sec)
            except Exception as e:
                logger.warning("[VLM-RUNTIME] snap 回调 update_polling_batch_interval 异常: %s", e)
            changed = True

        if changed:
            logger.info("[VLM-RUNTIME] 稳定区间 snap 回基础设置已完成")
        else:
            logger.info("[VLM-RUNTIME] 稳定区间检查：当前已接近基础设置，无需 snap (avg_latency=%.0fms)", avg_latency_ms)

    # ------------------------------------------------------------------ #
    #                       调整逻辑（减负载 / 加负载）                  #
    # ------------------------------------------------------------------ #

    def _adjust_down_load(self, *, avg_latency_ms: float) -> None:
        """
        降负载：
        - 增大 cut_window_sec；
        - 若为轮询模式，则增大 polling_batch_interval。
        """
        self._increase_cut_window(avg_latency_ms)

        if self.mode == MODEL.SECURITY_POLLING and self.update_polling_batch_interval:
            self._increase_polling_interval(avg_latency_ms)

    def _adjust_up_load(self, *, avg_latency_ms: float) -> None:
        """
        加负载：
        - 减小 cut_window_sec；
        - 若为轮询模式，则减小 polling_batch_interval。
        """
        self._decrease_cut_window(avg_latency_ms)

        if self.mode == MODEL.SECURITY_POLLING and self.update_polling_batch_interval:
            self._decrease_polling_interval(avg_latency_ms)

    # ------------------- cut_window_sec 调整 ------------------- #

    def _increase_cut_window(self, avg_latency_ms: float) -> None:
        cur = self.current_cut_window_sec
        if cur >= self.cut_window_max:
            return

        # panic 情况步子更大
        if avg_latency_ms >= self.panic_latency_ms:
            factor = 2.0
            delta = 6.0
        else:
            factor = 1.5
            delta = 3.0

        new_val = min(cur * factor, cur + delta, self.cut_window_max)
        if new_val <= cur + 1e-6:
            return

        self.current_cut_window_sec = new_val
        logger.warning(
            "[VLM-RUNTIME] 增大 cut_window_sec: %.1fs -> %.1fs (avg_latency=%.0fms)",
            cur, new_val, avg_latency_ms,
        )
        try:
            self.update_cut_window_sec(new_val)
        except Exception as e:
            logger.warning("[VLM-RUNTIME] 回调 update_cut_window_sec 异常: %s", e)

    def _decrease_cut_window(self, avg_latency_ms: float) -> None:
        cur = self.current_cut_window_sec
        if cur <= self.cut_window_min:
            return

        # 延迟很低可以多缩一点
        if avg_latency_ms <= self.low_latency_ms * 0.5:
            factor = 1.8
            delta = 3.0
        else:
            factor = 1.5
            delta = 2.0

        candidate1 = cur / factor
        candidate2 = cur - delta
        new_val = max(candidate1, candidate2, self.cut_window_min)
        if new_val >= cur - 1e-6:
            return

        self.current_cut_window_sec = new_val
        logger.info(
            "[VLM-RUNTIME] 减小 cut_window_sec: %.1fs -> %.1fs (avg_latency=%.0fms)",
            cur, new_val, avg_latency_ms,
        )
        try:
            self.update_cut_window_sec(new_val)
        except Exception as e:
            logger.warning("[VLM-RUNTIME] 回调 update_cut_window_sec 异常: %s", e)

    # ------------------- polling_interval 调整 ------------------- #

    def _increase_polling_interval(self, avg_latency_ms: float) -> None:
        """
        降负载 → 增大轮询间隔。
        注意：
        - interval 在“小范围区间”内（默认 <=3600s）时，步子稍大；
        - interval 大于该阈值时，用固定比例 +12% / +25% 增长，
          且只在最后做一次硬上限裁剪，不再用 min(...) 三选一避免“调反”。
        """
        cur = self.current_polling_interval_sec
        if cur <= 0:
            return

        # panic 情况稍微激进一点
        panic = avg_latency_ms >= self.panic_latency_ms

        if cur <= self._small_interval_hi:
            # “小~中等间隔”：加法 + 乘法结合，步子相对大一些
            if panic:
                step = max(cur * 0.6, 60.0)  # 至少 +60s
            else:
                step = max(cur * 0.3, 20.0)  # 至少 +20s
            new_val = cur + step
        else:
            # “大间隔”：用比例增长，避免一下子跑太远
            factor = 1.25 if panic else 1.12
            new_val = cur * factor

        # 最后统一做一次硬上限裁剪，避免反向
        new_val = min(new_val, self._hard_max_polling_interval)
        if new_val <= cur + 1e-6:
            return

        self.current_polling_interval_sec = new_val
        logger.warning(
            "[VLM-RUNTIME] 增大 polling_batch_interval: %.1fs -> %.1fs (avg_latency=%.0fms)",
            cur, new_val, avg_latency_ms,
        )
        try:
            self.update_polling_batch_interval(new_val)
        except Exception as e:
            logger.warning("[VLM-RUNTIME] 回调 update_polling_batch_interval 异常: %s", e)

    def _decrease_polling_interval(self, avg_latency_ms: float) -> None:
        """
        加负载 → 减小轮询间隔。
        - 小间隔时不再继续减到离谱；
        - 大间隔时用固定比例 -8% / -12%，最后统一裁剪到硬下限。
        """
        cur = self.current_polling_interval_sec
        if cur <= self._hard_min_polling_interval:
            return

        very_low = avg_latency_ms <= self.low_latency_ms * 0.4

        if cur <= self._small_interval_hi:
            # 已经不算太大了，只在“高于安全下限不少”时略微减一点
            if cur <= self._small_interval_lo * 1.2:
                # 离硬下限太近，不再减
                return
            # 这里用简单减法，避免抖动
            step = max(cur * 0.2, 15.0)  # 至少 15s
            new_val = cur - step
        else:
            # 大间隔，用比例缩小
            factor = 0.88 if very_low else 0.92
            new_val = cur * factor

        new_val = max(new_val, self._hard_min_polling_interval)
        if new_val >= cur - 1e-6:
            return

        self.current_polling_interval_sec = new_val
        logger.info(
            "[VLM-RUNTIME] 减小 polling_batch_interval: %.1fs -> %.1fs (avg_latency=%.0fms)",
            cur, new_val, avg_latency_ms,
        )
        try:
            self.update_polling_batch_interval(new_val)
        except Exception as e:
            logger.warning("[VLM-RUNTIME] 回调 update_polling_batch_interval 异常: %s", e)
