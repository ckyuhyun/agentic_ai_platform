import logging
import os
from logging.handlers import RotatingFileHandler
from pathlib import Path

# agentic_ai_platform/utils/logging_config.py -> agentic_ai_platform/logs
DEFAULT_LOG_DIR = Path(__file__).resolve().parents[1] / "logs"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

_configured = False


def setup_logging(log_dir: str | Path | None = None, level: int = logging.INFO) -> None:
    """
    Configure the root logger once for the whole app: a console handler plus
    one rotating file per level (info.log, warning.log, error.log), each
    cumulative (e.g. error.log only gets ERROR+, info.log gets everything).

    Safe to call from multiple package __init__.py files -- only the first
    call takes effect.
    """
    global _configured
    if _configured:
        return

    log_dir = Path(log_dir or os.getenv("LOG_DIR", DEFAULT_LOG_DIR))
    log_dir.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    file_levels = {
        "info.log": logging.INFO,
        "warning.log": logging.WARNING,
        "error.log": logging.ERROR,
    }
    for filename, file_level in file_levels.items():
        handler = RotatingFileHandler(
            log_dir / filename, maxBytes=10_000_000, backupCount=5, encoding="utf-8"
        )
        handler.setLevel(file_level)
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)

    _configured = True
