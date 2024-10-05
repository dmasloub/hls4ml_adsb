# src/utils/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler
from src.config.config import Config
from typing import Optional


class Logger:
    """
    A Logger class to manage multiple loggers, each writing to its own log file.
    """
    _loggers = {}

    @staticmethod
    def get_logger(name: str = __name__, log_filename: Optional[str] = None) -> logging.Logger:
        """
        Retrieves a logger instance. If it doesn't exist, creates a new one with specified configurations.

        Args:
            name (str, optional): Name of the logger. Defaults to __name__.
            log_filename (str, optional): Specific log file name for this logger.
                If None, defaults to "{sanitized_name}.log" in the logs directory.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if name in Logger._loggers:
            return Logger._loggers[name]

        # Initialize configuration to access log directory
        config = Config()

        # Create logs directory if it doesn't exist
        log_dir = config.paths.logs_dir
        os.makedirs(log_dir, exist_ok=True)

        # Define log file path
        if log_filename is None:
            # Sanitize logger name to create a valid file name
            sanitized_name = name.replace('.', '_').replace('/', '_')
            log_filename = f"{sanitized_name}.log"
        log_file = os.path.join(log_dir, log_filename)

        # Create a custom logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Capture all levels

        # Define log format
        formatter = logging.Formatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console handler for INFO and higher
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)

        # File handler for DEBUG and higher with rotation
        file_handler = RotatingFileHandler(
            log_file,
            mode='a',
            maxBytes=5 * 1024 * 1024,  # 5 MB
            backupCount=5,
            encoding='utf-8',
            delay=0
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)

        # Avoid adding multiple handlers if logger already has handlers
        if not logger.handlers:
            logger.addHandler(console_handler)
            logger.addHandler(file_handler)

        # Store the logger in the dictionary
        Logger._loggers[name] = logger

        return logger
