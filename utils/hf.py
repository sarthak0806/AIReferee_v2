"""Load Hugging Face models from the local cache first.

Checking the Hub on every start is slow, prints an "unauthenticated requests"
warning and fails offline. Models are only downloaded when they are missing.
"""
from __future__ import annotations

import logging
from typing import Callable, TypeVar

from utils.logging_config import get_logger

log = get_logger("hf")
T = TypeVar("T")

_quiet = False


def quiet_hf() -> None:
    global _quiet
    if _quiet:
        return
    try:
        from transformers.utils import logging as tlog
        tlog.set_verbosity_error()
        tlog.disable_progress_bar()          # "Loading weights: 100%|..." bars
    except Exception:
        pass
    try:
        from huggingface_hub.utils import logging as hlog
        hlog.set_verbosity_error()
    except Exception:
        pass
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)
    _quiet = True


def load_cached(factory: Callable[..., T], name: str, **kwargs) -> T:
    """``factory(name, local_files_only=True, ...)``, falling back to a download."""
    quiet_hf()
    try:
        return factory(name, local_files_only=True, **kwargs)
    except Exception as exc:
        log.info("%s not in local cache (%s); downloading", name, type(exc).__name__)
        return factory(name, **kwargs)
