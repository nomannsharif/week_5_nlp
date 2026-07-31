"""
utils/logger.py
===============
Centralized Logging Infrastructure for Training, Prediction, and Application events.
Generates dedicated log files in logs/ directory.
"""

import os
import logging
from config import TRAINING_LOG_PATH, PREDICTION_LOG_PATH, APPLICATION_LOG_PATH

def setup_logger(name: str, log_file: str, level=logging.INFO) -> logging.Logger:
    """Configures and returns a logger instance writing to specified log file and console."""
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Prevent duplicate handlers if re-initialized
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        # Stream/Console Handler
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

# Dedicated Centralized Loggers
training_logger = setup_logger("training", TRAINING_LOG_PATH)
prediction_logger = setup_logger("prediction", PREDICTION_LOG_PATH)
app_logger = setup_logger("application", APPLICATION_LOG_PATH)
