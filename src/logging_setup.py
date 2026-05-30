"""
src/logging_setup.py
====================
Centralized logging configuration. All modules use the standard
`logging` package; this module wires up a console handler plus an
optional rotating file handler so production logs don't fill the disk.
"""
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Optional


_FMT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"


def setup_logging(level: str = "INFO", log_file: Optional[str] = None) -> None:
    """Configure the root logger.

    Args:
        level:    Standard logging level name ("DEBUG", "INFO", ...).
        log_file: Optional path. When set, logs are also written to a
                  RotatingFileHandler (5 MB × 3 files). Empty string or
                  None disables file logging.
    """
    numeric_level = getattr(logging, level.upper(), logging.INFO)

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Clear any pre-existing handlers — re-running setup_logging in the
    # same process (e.g. during tests) otherwise duplicates log lines.
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = logging.Formatter(_FMT, datefmt=_DATEFMT)

    console = logging.StreamHandler()
    console.setLevel(numeric_level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if log_file:
        path = Path(log_file)
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            file_handler = RotatingFileHandler(
                path, maxBytes=5_000_000, backupCount=3, encoding="utf-8"
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            root.addHandler(file_handler)
        except OSError as exc:
            # File logging is a convenience, not a requirement — fall
            # back to console-only rather than crashing the app.
            root.warning("Could not open log file %s (%s); console-only.", path, exc)

    # Quiet down libraries that spam INFO during init.
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
