import logging
import sys
from pathlib import Path


def get_logger(name: str) -> logging.Logger:
    """
    Central logger for the ASRS early warning system.
    Writes to console (stdout) and outputs/logs/asrs_ew.log.
    Format: timestamp | level | module | message
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger  # already configured — avoid duplicate handlers on re-import

    logger.setLevel(logging.INFO)

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    log_dir = Path("outputs/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "asrs_ew.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger
