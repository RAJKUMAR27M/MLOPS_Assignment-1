"""
Logging Configuration for Heart Disease MLOps Project

Provides structured logging with JSON format for production environments.
"""

import logging
import sys
import os
from datetime import datetime
from pathlib import Path
from typing import Optional
import json

from pythonjsonlogger import jsonlogger


def get_project_root() -> Path:
    """Get the project root directory."""
    return Path(__file__).parent.parent.parent


class CustomJsonFormatter(jsonlogger.JsonFormatter):
    """
    Custom JSON formatter for structured logging.
    """

    def add_fields(self, log_record, record, message_dict):
        super(CustomJsonFormatter, self).add_fields(log_record, record, message_dict)

        # Add timestamp
        if not log_record.get("timestamp"):
            log_record["timestamp"] = datetime.utcnow().isoformat()

        # Add log level
        if log_record.get("level"):
            log_record["level"] = log_record["level"].upper()
        else:
            log_record["level"] = record.levelname

        # Add module and function
        log_record["module"] = record.module
        log_record["function"] = record.funcName
        log_record["line"] = record.lineno


def setup_logging(
    log_level: str = "INFO", log_format: str = "json", log_file: Optional[Path] = None
) -> logging.Logger:
    """
    Set up logging configuration.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Log format ('json' or 'text')
        log_file: Optional path to log file

    Returns:
        Configured logger
    """
    # Get log level from environment or parameter
    level = os.environ.get("LOG_LEVEL", log_level).upper()

    # Create logger
    logger = logging.getLogger("heart_disease_api")
    logger.setLevel(getattr(logging, level))

    # Remove existing handlers
    logger.handlers = []

    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(getattr(logging, level))

    if log_format == "json":
        formatter = CustomJsonFormatter("%(timestamp)s %(level)s %(name)s %(message)s")
    else:
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )

    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Add file handler if specified
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(getattr(logging, level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "heart_disease_api") -> logging.Logger:
    """
    Get a logger with the specified name.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    return logging.getLogger(name)


class RequestLogger:
    """
    Logger for API requests with structured format.
    """

    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or get_logger()

    def log_request(
        self,
        method: str,
        path: str,
        status_code: int,
        latency: float,
        request_id: Optional[str] = None,
        extra: Optional[dict] = None,
    ):
        """
        Log an API request.

        Args:
            method: HTTP method
            path: Request path
            status_code: Response status code
            latency: Request latency in seconds
            request_id: Optional request ID for tracing
            extra: Additional fields to log
        """
        log_data = {
            "type": "request",
            "method": method,
            "path": path,
            "status_code": status_code,
            "latency_ms": round(latency * 1000, 2),
        }

        if request_id:
            log_data["request_id"] = request_id

        if extra:
            log_data.update(extra)

        if status_code >= 500:
            self.logger.error(json.dumps(log_data))
        elif status_code >= 400:
            self.logger.warning(json.dumps(log_data))
        else:
            self.logger.info(json.dumps(log_data))

    def log_prediction(
        self,
        prediction: int,
        probability: float,
        risk_level: str,
        latency: float,
        request_id: Optional[str] = None,
    ):
        """
        Log a prediction.

        Args:
            prediction: Model prediction (0 or 1)
            probability: Prediction probability
            risk_level: Risk level category
            latency: Prediction latency in seconds
            request_id: Optional request ID for tracing
        """
        log_data = {
            "type": "prediction",
            "prediction": prediction,
            "probability": round(probability, 4),
            "risk_level": risk_level,
            "latency_ms": round(latency * 1000, 2),
        }

        if request_id:
            log_data["request_id"] = request_id

        self.logger.info(json.dumps(log_data))


# Default logger setup
def init_logging():
    """Initialize default logging configuration."""
    log_level = os.environ.get("LOG_LEVEL", "INFO")
    log_format = os.environ.get("LOG_FORMAT", "json")

    project_root = get_project_root()
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)

    log_file = log_dir / f'api_{datetime.now().strftime("%Y%m%d")}.log'

    return setup_logging(log_level=log_level, log_format=log_format, log_file=log_file)


if __name__ == "__main__":
    # Test logging
    logger = init_logging()
    logger.info("Logging configuration test")
    logger.debug("Debug message")
    logger.warning("Warning message")
    logger.error("Error message")
