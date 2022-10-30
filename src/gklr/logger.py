"""GKLR logger module."""
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
    """Set the level of the logger.

    Args:
        level: The level desired for the logger.
    """
    main_logger.setLevel(level)


def logger_get_level() -> int:
    """Gets the level of the logger.
    
    Returns:
        The current level of the logger."""
    return main_logger.getEffectiveLevel()


def _get_console_handler() -> logging.StreamHandler:
    """Create a console handler for the logger.

    Returns:
        The console handler.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(default_formatter)
    return handler


def _get_file_handler(logPath: str,
                      fileName: str,
) -> logging.FileHandler:
    """Create a file handler for the logger.

    Args:
        logPath: The path to the log file.
        fileName: The name of the log file.

    Returns:
        The file handler instance.
    """
    handler = logging.FileHandler("{0}/{1}.log".format(logPath, fileName), 
        mode="w")
    handler.setFormatter(default_formatter)
    return handler


def get_default_logger(name: str,
                       level: int,
                       logPath: str = ".",
                       fileName: str = "gklr",
) -> logging.Logger:
    """Create a logger with a default configuration.

    Args:
        name: The name of the logger.
        level: The level of the logger.
        logPath: The path to the log file. Default: '.'.
        fileName: The name of the log file. Default: 'gklr'.

    Returns:
        The logger instance."""
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