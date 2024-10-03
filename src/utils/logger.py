# src/utils/logger.py

import logging
import os
from logging.handlers import RotatingFileHandler
from src.config.config import Config

class Logger:
    _logger_instance = None

    @staticmethod
    def get_logger(name: str = __name__) -> logging.Logger:
        """
        Static access method to get the logger instance.

        Args:
            name (str, optional): Name of the logger. Defaults to __name__.

        Returns:
            logging.Logger: Configured logger instance.
        """
        if Logger._logger_instance is None:
            Logger._initialize_logger()
        return logging.getLogger(name)

    @staticmethod
    def _initialize_logger():
        """
        Initializes the logger with handlers, formatters, and log levels.
        """
        # Initialize configuration to access log directory
        config = Config()

        # Create logs directory if it doesn't exist
        log_dir = config.paths.logs_dir
        os.makedirs(log_dir, exist_ok=True)

        # Define log file path
        log_file = os.path.join(log_dir, "optimization.log")

        # Create a custom logger
        logger = logging.getLogger()
        logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

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
            maxBytes=5*1024*1024,  # 5 MB
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

        Logger._logger_instance = logger
