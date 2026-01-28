import logging
import sys

from backend.core.config import settings


def setup_logging() -> logging.Logger:
    """
    Configure and return the application logger.

    Returns:
        Configured logger instance.
    """
    # Create logger
    logger = logging.getLogger("inyeon")

    # Set level based on debug setting
    level = logging.DEBUG if settings.debug else logging.INFO
    logger.setLevel(level)

    # Avoid duplicate handlers
    if logger.handlers:
        return logger

    # Console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(level)

    # Format
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


# Global logger instance
logger = setup_logging()
