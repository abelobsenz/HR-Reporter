from __future__ import annotations

import logging
import os
from typing import Optional


def setup_logging(level: Optional[str] = None) -> None:
    raw_level = level or os.getenv("HR_REPORT_LOG_LEVEL", "INFO")
    level_name = str(raw_level).strip().upper()
    log_level = getattr(logging, level_name, logging.INFO)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
