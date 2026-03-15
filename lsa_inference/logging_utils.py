"""Logging setup: console + timestamped file output."""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name, log_dir="logs"):
    """Create a logger that writes to both console and a timestamped file.

    Args:
        name: Logger/experiment name (used in filename).
        log_dir: Directory for log files.

    Returns:
        logger: Configured logging.Logger instance.
        log_path: Path to the log file.
    """
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"{name}_{timestamp}.log")

    logger = logging.getLogger(f"{name}_{timestamp}")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    # File handler — everything
    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logger.addHandler(fh)

    # Console handler — INFO and above
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)

    return logger, log_path
