from __future__ import annotations

import logging
import re
from typing import Optional


LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
PATH_AWARE_MODES = ("basename", "raw_path", "raw_plus_basename")
_PATH_SPLIT = re.compile(r"[\\/]+")


def setup_logging(level_name: str = "INFO") -> None:
    """Shared logger setup for NLP CLI scripts."""
    level = getattr(logging, str(level_name).upper(), logging.INFO)
    logging.basicConfig(level=level, format=LOG_FORMAT)


def safe_basename(path_or_name: str) -> str:
    text = (path_or_name or "").strip()
    if not text:
        return ""
    parts = _PATH_SPLIT.split(text)
    return parts[-1] if parts else text


def compose_path_aware_text(
    raw_path: Optional[str],
    filename: Optional[str],
    mode: str = "raw_plus_basename",
) -> str:
    """
    Compose model input text with optional path context.

    - basename: basename only
    - raw_path: raw path only
    - raw_plus_basename: "raw_path ||| basename"
    """
    mode = (mode or "raw_plus_basename").strip().lower()
    if mode not in PATH_AWARE_MODES:
        mode = "raw_plus_basename"

    raw = str(raw_path or "").strip()
    base_src = str(filename or "").strip() or raw
    base = safe_basename(base_src)

    if mode == "basename":
        return base or raw
    if mode == "raw_path":
        return raw or base

    if raw and base:
        return f"{raw} ||| {base}"
    return raw or base
