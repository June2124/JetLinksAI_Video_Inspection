'''
Author: 13594053100@163.com
Date: 2025-09-30 09:42:45
LastEditTime: 2025-11-19 18:18:56
'''

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler

# ===================== 彩色日志格式 =====================

COLOR_MAP = {
    "DEBUG": "\033[90m",     # 灰色
    "INFO": "\033[92m",      # 绿色
    "WARNING": "\033[93m",   # 黄色
    "ERROR": "\033[91m",     # 红色
    "CRITICAL": "\033[95m",  # 洋红色
}
RESET = "\033[0m"


class ColorFormatter(logging.Formatter):
    """带颜色的控制台日志格式化器"""

    def format(self, record: logging.LogRecord) -> str:
        color = COLOR_MAP.get(record.levelname, "")
        message = super().format(record)
        return f"{color}{message}{RESET}"


# ===================== 全局汇总文件 Handler =====================

# 所有通过 get_logger 创建的 logger 共用这个 handler，
# 实现“所有控制台日志汇总到一个文件”。
_GLOBAL_CONSOLE_FILE_HANDLER: logging.Handler | None = None


def _get_log_root(default_root: str | Path = "logs") -> Path:
    """
    日志根目录：
    - 默认: logs/
    - 也可通过环境变量 JETLINKS_LOG_ROOT 覆盖
    """
    env_root = os.getenv("JETLINKS_LOG_ROOT")
    root = Path(env_root) if env_root else Path(default_root)
    root.mkdir(parents=True, exist_ok=True)
    return root


def _get_global_console_file_handler(
    *,
    log_root: str | Path = "logs", # TODO 按照实际需求修改日志存放路径
    rotate: str = "time",          # 按时间滚动
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 20,
    when: str = "midnight",        # 每天 0 点轮转
    interval: int = 1,
    utc: bool = False,
) -> logging.Handler:
    """
    创建 / 复用一个全局 FileHandler，把所有 logger 的日志集中写入一个文件。

    文件命名格式示例（每天一个文件）:
        logs/2025-11-19_all_console.log
        logs/2025-11-20_all_console.log
        ...
    """
    global _GLOBAL_CONSOLE_FILE_HANDLER
    if _GLOBAL_CONSOLE_FILE_HANDLER is not None:
        return _GLOBAL_CONSOLE_FILE_HANDLER

    root = _get_log_root(log_root)

    # === 当前日期的日志文件名 ===
    today = datetime.now().strftime("%Y-%m-%d")
    file_path = root / f"{today}_all_console.log"

    # === 按时间滚动 ===
    if rotate == "time":
        fh: logging.Handler = TimedRotatingFileHandler(
            file_path,
            when=when,              # 每天滚动
            interval=interval,      # 1 天
            backupCount=backup_count,
            encoding="utf-8",
            utc=utc,
        )
    elif rotate == "size":
        fh = RotatingFileHandler(
            file_path, maxBytes=max_bytes,
            backupCount=backup_count, encoding="utf-8"
        )
    else:
        fh = logging.FileHandler(file_path, encoding="utf-8")

    fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    fh.setFormatter(logging.Formatter(fmt))

    _GLOBAL_CONSOLE_FILE_HANDLER = fh
    return fh



# ===================== 原有按模块单独文件（可选功能） =====================

def _sanitize_file_name(name: str) -> str:
    """
    把 logger 名转成相对安全的文件名，例如：
    'src.workers.worker_a_cut' -> 'src_workers_worker_a_cut.log'
    """
    base = name.replace(":", "_").replace("/", "_").replace("\\", "_")
    base = base.replace(" ", "_").replace(".", "_")
    if not base:
        base = "app"
    return f"{base}.log"


# ===================== 日志工具函数 =====================

def get_logger(
    name: str,
    level=logging.INFO,
    file_name: str | None = None,
    color_console: bool = True,
    *,
    log_root: str | Path = "logs",   # 日志根目录
    rotate: str = "time",            # "size" | "time" | "off"
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 10,
    when: str = "midnight",
    interval: int = 1,
    utc: bool = False
) -> logging.Logger:
    """
    获取独立 logger：
    - 控制台：彩色输出
    - 全局汇总文件：logs/all_console.log（所有 logger 共用）
    - 可选：当前 logger 自己再写一个独立文件（file_name 不为空时）

    Args:
        name (str): logger 名称
        level (int): 日志级别
        file_name (str|None):
            - None：不为该 logger 单独建文件，只写到全局 all_console.log
            - ""：同 None 效果
            - "xxx.log"：该 logger 额外写一个独立文件
        color_console (bool): 控制台是否彩色
        log_root (str|Path): 日志根目录，默认 logs/ 或环境变量 JETLINKS_LOG_ROOT
        rotate (str): "size" 按大小轮转；"time" 按时间轮转；"off" 不轮转
        max_bytes (int): 按大小轮转阈值（同时影响全局文件和独立文件）
        backup_count (int): 保留的日志文件数
        when (str): 时间轮转单位
        interval (int): 时间轮转间隔
        utc (bool): 是否按 UTC 时间轮转
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

        # ---------- 控制台 Handler ----------
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(ColorFormatter(fmt) if color_console else logging.Formatter(fmt))
        logger.addHandler(sh)

        # ---------- 全局汇总文件 Handler（关键：所有 logger 共用一个） ----------
        global_fh = _get_global_console_file_handler(
            log_root=log_root,
            rotate=rotate,
            max_bytes=max_bytes,
            backup_count=backup_count,
            when=when,
            interval=interval,
            utc=utc,
        )
        # 避免重复添加
        if global_fh not in logger.handlers:
            logger.addHandler(global_fh)

        # ---------- （可选）该 logger 自己的独立文件 ----------
        # 如果你不想要独立文件，可以让 file_name=None 或 ""。
        if file_name:
            # 如果传入的是 True / "auto"，就根据 name 生成
            if file_name is True or file_name == "auto":
                file_name = _sanitize_file_name(name)

            # 独立日志放在 log_root 子目录下，可以按日期分目录
            root = _get_log_root(log_root)
            date_dir = datetime.now().strftime("%Y-%m-%d")
            log_dir = root / date_dir
            log_dir.mkdir(parents=True, exist_ok=True)
            file_path = log_dir / file_name

            if rotate == "size":
                fh_individual: logging.Handler = RotatingFileHandler(
                    file_path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
                )
            elif rotate == "time":
                fh_individual = TimedRotatingFileHandler(
                    file_path, when=when, interval=interval,
                    backupCount=backup_count, encoding="utf-8", utc=utc
                )
            else:  # "off"
                fh_individual = logging.FileHandler(file_path, encoding="utf-8")

            fh_individual.setFormatter(logging.Formatter(fmt))
            logger.addHandler(fh_individual)

    # 关闭向 root 传播，避免被别的库重复处理
    logger.propagate = False
    return logger
