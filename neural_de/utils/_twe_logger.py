"""
This module provides a logger, adapted from the original Confiance logger, for logging messages
with specified formatting and output control. It can log messages to the standard output,
to a specified file, or both.

Usage:
    Import the module, then get the default logger:

    . code-block:: python

        import twe_logger
        logger = twe_logger.get_logger()

    If you need a logger with different parameters, call ``get_logger(.)`` with the desired
    parameters:

    . code-block:: python

        logger = twe_logger.get_logger(filename="my_logs.log") logger = twe_logger.get_logger(
        name="my_logger", level='debug',filename='my_logs.log', output="both")

    Then, use the logger within your code:

    . code-block:: python

        logger.info("This is an info message")
        logger.error("This is an error message")

Attributes:
    LOGGER_DEFAULT_NAME: Default name for the neural_de logger
    LOG_LEVEL: Logging level for all the methods. Options are ``"info"``, ``"debug"``,
        "warning"``, and ``"critical"``
    LOG_OUTPUT: Logging target. Should be ``"stdout"``, ``"file"``, or ``"both"``. If ``'file'`` or
        ``'both'``, ``LOG_FILE`` will be used as the destination file.
    LOG_FILE : Name of the file where the logger should write.
"""

import logging
import sys
from typing import Union

LOGGER_DEFAULT_NAME: str = 'neural_de_logger'
LOG_LEVEL: str = "debug"
LOG_OUTPUT: str = "stdout"
LOG_FILE: str = "log.csv"


def log_str_to_level(str_level: str) -> int:
    """
    Converts a string to a corresponding logging level.

    Args:
        str_level: The logging level as a string.

    Returns:
        The corresponding logging level.
    """
    if str_level == 'debug':
        level = logging.DEBUG
    elif str_level == 'info':
        level = logging.INFO
    elif str_level == 'warning':
        level = logging.WARNING
    elif str_level == 'error':
        level = logging.ERROR
    elif str_level == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.NOTSET
    return level


def get_logger(name: str = LOGGER_DEFAULT_NAME,
               level: Union[int, str] = LOG_LEVEL,
               filename: str = LOG_FILE,
               output: str = LOG_OUTPUT) -> logging.Logger:
    """
    Creates and returns a logger.

    Args:
        name: Optional, the name of the logger.
        level: Optional, the logging level.
        filename: Optional, name of the file where the logger should write.
        output: Optional, where should the logger write. Can be ``stdout``, ``file``, or ``both``.

    Returns:
      The logger.
    """
    if isinstance(level, str):
        level = log_str_to_level(level)

    # Decide the output based on whether a filename is provided and output is specified
    if output is None:
        output = 'file' if filename else 'stdout'

    logger = logging.getLogger(name)
    formatter = logging.Formatter(
        '[%(asctime)s] {%(pathname)s:%(lineno)d} %(levelname)s - %(message)s',
        '%m-%d %H:%M:%S')

    handlers = []

    if output in ['stdout', 'both']:
        # StreamHandler to send logs to stdout
        stream_handler = logging.StreamHandler(stream=sys.stdout)
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level)
        handlers.append(stream_handler)

    if output in ['file', 'both'] and filename:
        # FileHandler to send logs to a file
        file_handler = logging.FileHandler(filename)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(level)
        handlers.append(file_handler)

    if not logging.getLogger(name).hasHandlers():
        for handler in handlers:
            logger.addHandler(handler)
    else:
        logger.handlers = handlers
    logger.setLevel(level)
    logger.info("Logger: name: %s, handlers: %s", name, logger.handlers)

    return logger


def log_and_raise(logger: logging.Logger, exception: type[Exception], content: str) -> None:
    """
    Log an error message then raising an exception with the same message.

    Args:
      logger: Logger instance.
      exception: The exception we want to raise.
      content: Text content to log and add to the raised exception.
    Raises:
      ``exception`` : Raises the provided Exception with the provided message
    """
    logger.error(content)
    raise exception(content)
