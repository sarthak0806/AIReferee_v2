import logging
import os
import sys

from config import path

_CONFIGURED = False


def get_logger(name: str) -> logging.Logger:
    global _CONFIGURED
    if not _CONFIGURED:
        log_dir = path("log_dir")
        log_dir.mkdir(parents=True, exist_ok=True)
        fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s")
        root = logging.getLogger("aireferee")
        root.setLevel(logging.INFO)
        fh = logging.FileHandler(log_dir / "aireferee.log", encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        # Console verbosity: INFO for CLI progress; the Streamlit app sets WARNING
        # (its progress is shown in the UI). The log file always keeps INFO.
        sh.setLevel(os.environ.get("AIREFEREE_CONSOLE_LOG", "INFO").upper())
        root.addHandler(fh)
        root.addHandler(sh)
        _CONFIGURED = True
    return logging.getLogger(f"aireferee.{name}")
