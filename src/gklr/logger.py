"""GKLR logger module."""
from typing import Optional

import logging
import os

__all__ = [
    "logger_set_level",
    "logger_get_level",
    "logger_log",
    "logger_debug",
    "logger_info",
    "logger_warning",
    "logger_error",
    "logger_critical",
]

VERBOSITY_DEFAULT = logging.INFO

DEBUG = logging.DEBUG
INFO = logging.INFO
WARNING = logging.WARNING
ERROR = logging.ERROR
CRITICAL = logging.CRITICAL


def logger_set_level(level: int) -> None:
    """Set the level of the logger."""
    main_logger.setLevel(level)


def logger_get_level() -> int:
    """Gets the level of the logger."""
    return main_logger.getEffectiveLevel()


def _get_console_handler():
    handler = logging.StreamHandler()
    handler.setFormatter(default_formatter)
    return handler


def _get_file_handler(logPath, fileName):
    handler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName), 
        mode="w")
    handler.setFormatter(default_formatter)
    return handler


def get_default_logger(name: str, level: int, logPath: Optional[str] = ".", 
    fileName: Optional[str] = "gklr",
) -> logging.Logger:
    logger = logging.getLogger(name)
    #logger.addHandler(_get_console_handler())
    logger.addHandler(_get_file_handler(logPath, fileName))
    logger.setLevel(level)
    logger.propagate = False
    return logger


default_formatter = logging.Formatter(
    "[{asctime:s}] {levelname:s}: {message:s}",
    style="{",
    datefmt="%Y-%m-%d %H:%M:%S",
)

main_logger = get_default_logger("gklr", VERBOSITY_DEFAULT, os.getcwd())
logger_log = main_logger.log
logger_debug = main_logger.debug
logger_info = main_logger.info
logger_warning = main_logger.warning
logger_error = main_logger.error
logger_critical = main_logger.critical