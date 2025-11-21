'''
Author: 13594053100@163.com
Date: 2025-11-19 18:26:26
LastEditTime: 2025-11-19 18:26:30
'''

# -*- coding: utf-8 -*-
"""
Author: 13594053100@163.com
Date: 2025-11-19 18:40:00
Desc: 定期清理 A 侧产出的临时音视频及证据帧图片文件。
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Iterable, Optional

from src.utils.logger_utils import get_logger

logger = get_logger("file_cleanup")

def _safe_unlink(p: Path) -> bool:
    """安全删除单个文件，删除成功返回 True。"""
    try:
        if p.is_file() or p.is_symlink():
            p.unlink()
            logger.debug(f"[CLEANUP] 删除文件: {p}")
            return True
    except FileNotFoundError:
        return False
    except Exception as e:
        logger.warning(f"[CLEANUP] 删除文件失败: {p}, err={e}")
    return False


def _remove_empty_dirs(root: Path) -> None:
    """自底向上删除空目录。"""
    try:
        for dirpath, dirnames, filenames in os.walk(root, topdown=False):
            d = Path(dirpath)
            # 只清理 root 子目录，不去删 root 本身（防止把 out/ 或 evidence_images/ 删没）
            if d == root:
                continue
            if not dirnames and not filenames:
                try:
                    d.rmdir()
                    logger.debug(f"[CLEANUP] 删除空目录: {d}")
                except OSError:
                    # 可能被并发创建了新文件，忽略
                    pass
    except Exception as e:
        logger.warning(f"[CLEANUP] 清理空目录过程中异常: {e}")


def _cleanup_one_root(
    root: Path,
    *,
    ttl_seconds: int,
    exts: Optional[Iterable[str]] = None,
) -> int:
    """
    清理单个根目录 root 下“修改时间早于 ttl_seconds 的文件”。

    Args:
        root: 根目录
        ttl_seconds: 保存时长（秒），早于当前时间 - ttl_seconds 的文件会被删除
        exts: 需要匹配的扩展名列表（如 [".mp4", ".wav", ".jpg"]）。
              若为 None 或空，则不按后缀过滤，所有文件都会按时间判断。

    Returns:
        删除文件数量
    """
    if not root.exists() or not root.is_dir():
        return 0

    exts_norm = None
    if exts:
        exts_norm = {e.lower() for e in exts}

    now_ts = time.time()
    removed_count = 0

    for p in root.rglob("*"):
        if not p.is_file():
            continue

        # 后缀过滤（可选）
        if exts_norm is not None:
            if p.suffix.lower() not in exts_norm:
                continue

        try:
            mtime = p.stat().st_mtime
        except FileNotFoundError:
            continue
        except Exception as e:
            logger.debug(f"[CLEANUP] 无法获取文件时间: {p}, err={e}")
            continue

        if now_ts - mtime < ttl_seconds:
            # 还在保留期内
            continue

        if _safe_unlink(p):
            removed_count += 1

    # 尝试清理空目录
    _remove_empty_dirs(root)

    return removed_count


def run_cleanup_once(
    *,
    out_dir: str | Path,
    evidence_dir: str | Path,
    out_ttl_hours: int = 24,
    evidence_ttl_days: int = 7,
) -> None:
    """
    立即执行一次清理任务（可用于手动触发或定时任务内部调用）。

    Args:
        out_dir: A 侧临时切片目录（例如 "out"）
        evidence_dir: 证据帧目录（例如 "JetLinksAI/static/evidence_images"）
        out_ttl_hours: out 目录保留时长（小时）
        evidence_ttl_days: 证据帧保留时长（天）
    """
    out_path = Path(out_dir).absolute()
    ev_path = Path(evidence_dir).absolute()

    out_ttl = max(1, int(out_ttl_hours)) * 3600
    ev_ttl = max(1, int(evidence_ttl_days)) * 86400

    logger.info(
        "[CLEANUP] 开始执行一次清理: out='%s'(TTL=%dh), evidence='%s'(TTL=%dd)",
        out_path, out_ttl_hours, ev_path, evidence_ttl_days,
    )

    removed_out = _cleanup_one_root(
        out_path,
        ttl_seconds=out_ttl,
        # out 下通常是 mp4/wav/jpg 这类临时文件，你也可以扩展
        exts=[".mp4", ".wav", ".jpg", ".jpeg", ".png"],
    )

    removed_ev = _cleanup_one_root(
        ev_path,
        ttl_seconds=ev_ttl,
        # 证据帧通常是图片
        exts=[".jpg", ".jpeg", ".png"],
    )

    logger.info(
        "[CLEANUP] 本次清理完成: out 删除 %d 个文件, evidence 删除 %d 个文件",
        removed_out, removed_ev,
    )


def start_cleanup_daemon(
    *,
    out_dir: str | Path,
    evidence_dir: str | Path,
    out_ttl_hours: int = 24,
    evidence_ttl_days: int = 7,
    interval_hours: int = 1,
) -> threading.Thread:
    """
    启动一个后台守护线程，定期清理临时文件。

    Args:
        out_dir: A 侧临时切片目录（如 "out"）
        evidence_dir: 证据帧目录（如 "JetLinksAI/static/evidence_images"）
        out_ttl_hours: out 目录保留时长（小时）
        evidence_ttl_days: 证据帧保留时长（天）
        interval_hours: 清理任务执行周期（小时）

    Returns:
        后台线程对象（daemon=True）
    """
    out_path = Path(out_dir).absolute()
    ev_path = Path(evidence_dir).absolute()
    interval_sec = max(1, int(interval_hours)) * 3600

    def _loop() -> None:
        logger.info(
            "[CLEANUP] 清理守护线程启动: out='%s'(TTL=%dh), evidence='%s'(TTL=%dd), interval=%dh",
            out_path, out_ttl_hours, ev_path, evidence_ttl_days, interval_hours,
        )
        while True:
            try:
                run_cleanup_once(
                    out_dir=out_path,
                    evidence_dir=ev_path,
                    out_ttl_hours=out_ttl_hours,
                    evidence_ttl_days=evidence_ttl_days,
                )
            except Exception as e:
                logger.error(f"[CLEANUP] 定期清理任务异常: {e}")
            # 睡到下一轮
            time.sleep(interval_sec)

    t = threading.Thread(target=_loop, name="FileCleanupDaemon", daemon=True)
    t.start()
    return t
