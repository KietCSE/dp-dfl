"""Centralized logger configuration for DPFL experiments.

Usage:
    from dpfl.tracking.logger_setup import setup_experiment_logger
    logger = setup_experiment_logger(run_dir)
"""

import logging
import os


# Console format: compact, no timestamp (matches existing print style)
CONSOLE_FMT = "%(message)s"

# File format: full detail with timestamp + module
FILE_FMT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
FILE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_experiment_logger(run_dir: str, log_filename: str = "experiment.log") -> logging.Logger:
    """Configure root 'dpfl' logger with file + console handlers.

    Args:
        run_dir: Output directory (e.g. results/dpsgd_kurtosis_20260412_112340/)
        log_filename: Log file name inside run_dir

    Returns:
        Configured logger instance for 'dpfl' namespace.
    """
    logger = logging.getLogger("dpfl")

    # Avoid duplicate handlers on repeated calls
    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    # Don't propagate to root — opacus installs a StreamHandler on root,
    # which would duplicate every log line on console.
    logger.propagate = False

    # File handler: ALL levels (DEBUG+)
    os.makedirs(run_dir, exist_ok=True)
    log_path = os.path.join(run_dir, log_filename)
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(FILE_FMT, datefmt=FILE_DATEFMT))
    logger.addHandler(fh)

    # Console handler: INFO+
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(CONSOLE_FMT))
    logger.addHandler(ch)

    return logger


def setup_batch_logger(log_dir: str, log_filename: str = "batch.log") -> logging.Logger:
    """Configure logger for batch runner.

    Args:
        log_dir: Directory for batch log (e.g. results/experiments/)
        log_filename: Log file name

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("dpfl.batch")

    if logger.handlers:
        return logger

    logger.setLevel(logging.DEBUG)
    logger.propagate = False  # don't hit opacus's root handler

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_filename)
    fh = logging.FileHandler(log_path, mode="a", encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(FILE_FMT, datefmt=FILE_DATEFMT))
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(logging.Formatter(CONSOLE_FMT))
    logger.addHandler(ch)

    return logger
