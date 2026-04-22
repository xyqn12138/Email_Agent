"""项目日志工具，统一管理日志输出。"""

from __future__ import annotations

import logging
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from agent.utils.path_handler import get_absolute_path

LOGGER_NAME = "email_agent"
LOG_LEVEL = logging.INFO
LOG_DIR = Path(get_absolute_path(r"src\log"))
LOG_FILE = LOG_DIR / "email_agent.log"


def _build_formatter() -> logging.Formatter:
	"""创建统一日志格式。"""
	return logging.Formatter(
		fmt="%(asctime)s | %(levelname)s | %(name)s | %(filename)s:%(lineno)d | %(message)s",
		datefmt="%Y-%m-%d %H:%M:%S",
	)


def setup_logger(name: str = LOGGER_NAME) -> logging.Logger:
	"""初始化并返回项目 logger。

	仅初始化一次；后续重复调用将直接返回已配置 logger。
	日志级别固定为 INFO，输出到控制台和 log/email_agent.log。
	"""
	logger = logging.getLogger(name)
	if logger.handlers:
		return logger

	LOG_DIR.mkdir(parents=True, exist_ok=True)

	logger.setLevel(LOG_LEVEL)
	logger.propagate = False
	formatter = _build_formatter()

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(LOG_LEVEL)
	stream_handler.setFormatter(formatter)

	file_handler = TimedRotatingFileHandler(
		filename=LOG_FILE,
		when="midnight",
		interval=1,
		backupCount=7,
		encoding="utf-8",
	)
	file_handler.setLevel(LOG_LEVEL)
	file_handler.setFormatter(formatter)

	logger.addHandler(stream_handler)
	logger.addHandler(file_handler)

	return logger


def get_logger(name: str = LOGGER_NAME) -> logging.Logger:
	"""获取可直接使用的 logger。"""
	return setup_logger(name)


if __name__ == "__main__":
    logger = get_logger()
    logger.info("日志系统初始化成功！")


