#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 1: Dataset preparation and ChatML formatting.

This script:
1. Loads source JSON records (default: parsed_dataset.json)
2. Converts samples to Qwen ChatML messages (system/user/assistant)
3. Uses basename-aware user input composition
4. Splits data into train/validation and saves JSONL outputs
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nlp.shared import PATH_AWARE_MODES, compose_path_aware_text


LOGGER = logging.getLogger("qwen_sft_data_prep")

# Fixed field order to keep output JSON schema stable.
FIELDS = [
    "title",
    "zh_title",
    "year",
    "season_episode",
    "resolution",
    "frame_rate",
    "source",
    "video_codec",
    "video_hdr",
    "audio_codec",
    "group",
]

SYSTEM_PROMPT = (
    "你是一个专业的影视文件解析引擎。请提取用户输入的文件名中的实体信息，并严格以JSON格式输出。"
    "包含字段：title, zh_title, year, season_episode, resolution, frame_rate, source, video_codec, video_hdr, audio_codec, group。"
    "缺失的字段输出null。将季集数规范化为SxxExx格式。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare Qwen SFT dataset from filename-parser JSON."
    )
    parser.add_argument("--input_json", type=str, default="parsed_dataset.json")
    parser.add_argument("--output_dir", type=str, default="data/qwen_sft_parser")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--path_aware_mode",
        type=str,
        default="raw_plus_basename",
        choices=PATH_AWARE_MODES,
        help="How user input text is composed in ChatML samples.",
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=-1.0,
        help="Minimum confidence threshold; use -1.0 to disable filtering.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Optional cap for quick validation runs.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite target directory if it exists; otherwise create a versioned directory.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def ensure_output_dir(base_dir: Path, overwrite: bool) -> Path:
    """
    Determine output directory policy:
    - overwrite=True: reuse existing directory
    - overwrite=False: append suffixes such as _v2 / _v3 when needed
    """
    if overwrite:
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    if not base_dir.exists():
        base_dir.mkdir(parents=True, exist_ok=True)
        return base_dir

    parent = base_dir.parent
    stem = base_dir.name
    idx = 2
    while True:
        candidate = parent / f"{stem}_v{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        idx += 1


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input json not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON root must be a list.")
    return data


def safe_basename(raw: str) -> str:
    """
    Return basename from either Windows or Unix style paths.
    """
    text = (raw or "").strip()
    if not text:
        return ""
    parts = re.split(r"[\\/]+", text)
    return parts[-1] if parts else text


def normalize_season_episode(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    # SxxExx
    m = re.search(r"(?i)s\s*0*(\d{1,3})\s*e\s*0*(\d{1,4})", text)
    if m:
        season = int(m.group(1))
        episode = int(m.group(2))
        return f"S{season:02d}E{episode:02d}"

    # 3x12
    m = re.search(r"(?i)\b0*(\d{1,3})\s*x\s*0*(\d{1,4})\b", text)
    if m:
        season = int(m.group(1))
        episode = int(m.group(2))
        return f"S{season:02d}E{episode:02d}"

    # E12 -> S01E12
    m = re.search(r"(?i)\be\s*0*(\d{1,4})\b", text)
    if m:
        episode = int(m.group(1))
        return f"S01E{episode:02d}"

    # Numeric-only values are interpreted as episode numbers in season 1.
    if re.fullmatch(r"\d{1,4}", text):
        episode = int(text)
        return f"S01E{episode:02d}"

    # Keep the original value when no supported normalization pattern matches.
    return text


def normalize_frame_rate(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    m = re.search(r"(?i)\b(\d{2,3}(?:\.\d{1,3})?)\s*fps\b", text)
    if m:
        return f"{m.group(1)}fps"

    m = re.fullmatch(r"(\d{2,3}(?:\.\d{1,3})?)", text)
    if m:
        return f"{m.group(1)}fps"

    return text


def build_target_json(parsed: Dict[str, Any]) -> Dict[str, Any]:
    target: Dict[str, Any] = {}
    for field in FIELDS:
        value = parsed.get(field)
        if field == "season_episode":
            target[field] = normalize_season_episode(value)
        elif field == "frame_rate":
            target[field] = normalize_frame_rate(value)
        else:
            target[field] = value if value is not None else None
    return target


def build_chat_sample(
    record: Dict[str, Any], path_aware_mode: str
) -> Optional[Dict[str, Any]]:
    parsed = record.get("parsed") if isinstance(record.get("parsed"), dict) else {}

    source_text = record.get("raw_path") or record.get("filename") or ""
    filename = safe_basename(str(source_text))
    if not filename:
        return None

    model_input_text = compose_path_aware_text(
        raw_path=str(record.get("raw_path") or ""),
        filename=str(record.get("filename") or filename),
        mode=path_aware_mode,
    )
    if not model_input_text:
        return None

    target = build_target_json(parsed)
    assistant_output = json.dumps(target, ensure_ascii=False)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"?????{model_input_text}"},
        {"role": "assistant", "content": assistant_output},
    ]

    return {
        "messages": messages,
        "filename": filename,
        "model_input_text": model_input_text,
        "target_json": target,
    }


def split_train_valid(
    samples: List[Dict[str, Any]], train_ratio: float, seed: int
) -> Tuple[List, List]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1).")
    rng = random.Random(seed)
    shuffled = list(samples)
    rng.shuffle(shuffled)
    split_idx = int(len(shuffled) * train_ratio)
    split_idx = max(1, min(split_idx, len(shuffled) - 1))
    return shuffled[:split_idx], shuffled[split_idx:]


def save_jsonl(rows: List[Dict[str, Any]], path: Path) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    args = parse_args()
    setup_logging()

    input_json = Path(args.input_json)
    out_dir = ensure_output_dir(Path(args.output_dir), overwrite=args.overwrite)

    records = load_records(input_json)
    LOGGER.info("Loaded %d raw records from %s", len(records), input_json)

    samples: List[Dict[str, Any]] = []
    skipped_low_conf = 0
    skipped_invalid = 0

    for record in records:
        confidence = float(record.get("confidence", 0.0) or 0.0)
        if confidence < args.min_confidence:
            skipped_low_conf += 1
            continue

        sample = build_chat_sample(record, path_aware_mode=args.path_aware_mode)
        if sample is None:
            skipped_invalid += 1
            continue
        samples.append(sample)

    if args.max_samples is not None and args.max_samples > 0:
        samples = samples[: args.max_samples]

    if len(samples) < 2:
        raise RuntimeError("可用样本不足，无法切分 train/validation。")

    train_rows, valid_rows = split_train_valid(
        samples, train_ratio=args.train_ratio, seed=args.seed
    )
    train_path = out_dir / "train.jsonl"
    valid_path = out_dir / "validation.jsonl"

    save_jsonl(train_rows, train_path)
    save_jsonl(valid_rows, valid_path)

    meta = {
        "input_json": str(input_json),
        "output_dir": str(out_dir),
        "train_size": len(train_rows),
        "validation_size": len(valid_rows),
        "total_kept": len(samples),
        "skipped_low_confidence": skipped_low_conf,
        "skipped_invalid": skipped_invalid,
        "system_prompt": SYSTEM_PROMPT,
        "fields": FIELDS,
        "seed": args.seed,
        "train_ratio": args.train_ratio,
        "min_confidence": args.min_confidence,
    }
    with (out_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    LOGGER.info(
        "Done. train=%d valid=%d output=%s", len(train_rows), len(valid_rows), out_dir
    )


if __name__ == "__main__":
    main()
