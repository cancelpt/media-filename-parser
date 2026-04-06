#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compare NER predictions against rule-engine parsed labels where confidence == 1.0.

This script focuses on field-level agreement and explicitly normalizes
`season_episode` to a canonical form before matching.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlp.shared import compose_path_aware_text, setup_logging as setup_shared_logging
from nlp.ner.train_media_filename_ner import (
    PARSED_KEY_TO_ENTITY,
    MediaFilenameNERPredictor,
)


LOGGER = logging.getLogger("ner_vs_rule_conf1_eval")
FIELDS: List[str] = list(PARSED_KEY_TO_ENTITY.keys())
SEASON_EP_FIELD = "season_episode"
YEAR_PATTERN = re.compile(r"(19\d{2}|20[0-3]\d)")
FPS_PATTERN = re.compile(r"(?i)\b(\d{2,3}(?:\.\d{1,3})?)\s*fps\b")
RANGE_PATTERN = re.compile(r"^\s*(\d{1,4})\s*[-~]\s*(\d{1,4})\s*$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NER vs rule-engine labels where confidence equals 1.0."
    )
    parser.add_argument("--dataset_path", type=str, default="parsed_dataset.json")
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/media_filename_ner_crf_pathaware_20260402_213115",
    )
    parser.add_argument(
        "--output_dir", type=str, default="outputs/ner_vs_rule_conf1_eval"
    )
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument(
        "--device", type=str, default=None, help="cuda/cpu, default auto."
    )
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Optional cap for quick debug."
    )
    parser.add_argument("--log_every", type=int, default=500)
    parser.add_argument(
        "--path_aware_mode",
        type=str,
        default="raw_plus_basename",
        choices=("basename", "raw_path", "raw_plus_basename"),
    )
    parser.add_argument(
        "--conf_target",
        type=float,
        default=1.0,
        help="Rule confidence target value. Default compares confidence == 1.0.",
    )
    parser.add_argument(
        "--conf_eps",
        type=float,
        default=1e-9,
        help="Floating tolerance for confidence compare.",
    )
    parser.add_argument("--keep_mismatch_samples", type=int, default=200)
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def normalize_text_basic(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = unicodedata.normalize("NFKC", str(value)).strip()
    if not text:
        return None
    text = re.sub(r"\s+", " ", text)
    return text.lower()


def parse_cjk_number(token: str) -> Optional[int]:
    if token is None:
        return None
    token = unicodedata.normalize("NFKC", str(token)).strip()
    if not token:
        return None
    if token.isdigit():
        return int(token)

    digits = {
        "零": 0,
        "〇": 0,
        "一": 1,
        "二": 2,
        "两": 2,
        "三": 3,
        "四": 4,
        "五": 5,
        "六": 6,
        "七": 7,
        "八": 8,
        "九": 9,
    }
    units = {"十": 10, "百": 100}

    total = 0
    current = 0
    seen = False
    for ch in token:
        if ch in digits:
            current = digits[ch]
            seen = True
            continue
        if ch in units:
            seen = True
            unit = units[ch]
            if current == 0:
                current = 1
            total += current * unit
            current = 0
            continue
        return None
    total += current
    if not seen:
        return None
    return total


def canonicalize_season_episode(value: Any) -> Optional[str]:
    """
    Canonical format for robust matching:
      - SxEy, Sx, Ey, Dx, SxDy, Rstart-end, SPECIAL_TAG
      - pure number => Ey
    """
    if value is None:
        return None
    raw = unicodedata.normalize("NFKC", str(value)).strip()
    if not raw:
        return None

    text = raw.upper()
    text = text.replace("—", "-").replace("–", "-").replace("~", "-")
    compact = re.sub(r"\s+", "", text)

    cn_season = re.search(r"第([0-9一二两三四五六七八九十百零〇]+)季", raw)
    cn_episode = re.search(r"第([0-9一二两三四五六七八九十百零〇]+)[集话話]", raw)
    cn_season_inline_ep = re.search(
        r"第([0-9一二两三四五六七八九十百零〇]+)季\s*[-_ ]+\s*(\d{1,4})(?!\d)",
        raw,
    )
    if cn_season_inline_ep:
        s_num = parse_cjk_number(cn_season_inline_ep.group(1))
        e_num = int(cn_season_inline_ep.group(2))
        if s_num is not None:
            return f"S{s_num}E{e_num}"

    if cn_season or cn_episode:
        s_num = parse_cjk_number(cn_season.group(1)) if cn_season else None
        e_num = parse_cjk_number(cn_episode.group(1)) if cn_episode else None
        if s_num is not None and e_num is not None:
            return f"S{s_num}E{e_num}"
        if s_num is not None:
            return f"S{s_num}"
        if e_num is not None:
            return f"E{e_num}"

    m = re.search(r"S(\d{1,3})E[P]?(\d{1,4})(?!\d)", compact)
    if m:
        return f"S{int(m.group(1))}E{int(m.group(2))}"

    m = re.search(r"(\d{1,3})X(\d{1,4})(?!\d)", compact)
    if m:
        return f"S{int(m.group(1))}E{int(m.group(2))}"

    m = re.search(r"S(\d{1,3})D(\d{1,4})(?!\d)", compact)
    if m:
        return f"S{int(m.group(1))}D{int(m.group(2))}"

    m = re.fullmatch(r"S(\d{1,3})", compact)
    if m:
        return f"S{int(m.group(1))}"

    m = re.search(r"S(\d{1,3})[-_](\d{1,4})(?!\d)", compact)
    if m:
        return f"S{int(m.group(1))}E{int(m.group(2))}"

    m = re.search(r"SEASON(\d{1,3})[-_](\d{1,4})(?!\d)", compact)
    if m:
        return f"S{int(m.group(1))}E{int(m.group(2))}"

    m = re.fullmatch(r"EP?(\d{1,4})", compact)
    if m:
        return f"E{int(m.group(1))}"

    m = re.fullmatch(r"D(\d{1,4})", compact)
    if m:
        return f"D{int(m.group(1))}"

    m = RANGE_PATTERN.fullmatch(compact)
    if m:
        return f"R{int(m.group(1))}-{int(m.group(2))}"

    if re.fullmatch(r"\d{1,4}", compact):
        return f"E{int(compact)}"

    if re.fullmatch(r"(?:SP|OVA|OAD|BONUS|EXTRAS?|COMPLETE)(?:[-_ ]?\d+)?", compact):
        return compact.replace("_", "").replace(" ", "")

    return compact


def normalize_value(field: str, value: Any) -> Optional[str]:
    if field == SEASON_EP_FIELD:
        return canonicalize_season_episode(value)

    basic = normalize_text_basic(value)
    if basic is None:
        return None

    if field == "year":
        m = YEAR_PATTERN.search(basic)
        return m.group(1) if m else basic

    if field == "resolution":
        return basic.replace(" ", "").upper()

    if field == "frame_rate":
        m = FPS_PATTERN.search(basic)
        if m:
            return f"{m.group(1)}fps"
        just_num = re.fullmatch(r"(\d{2,3}(?:\.\d{1,3})?)", basic)
        if just_num:
            return f"{just_num.group(1)}fps"
        return basic.replace(" ", "")

    # For codec/source/group/title fields, keep spaces but compare case-insensitively.
    return basic


def normalize_value_raw_for_se(value: Any) -> Optional[str]:
    # Raw season_episode compare baseline (without canonical conversion).
    return normalize_text_basic(value)


def is_confidence_match(value: Any, target: float, eps: float) -> bool:
    try:
        return abs(float(value) - target) <= eps
    except Exception:  # pylint: disable=broad-except
        return False


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a JSON list.")
    return data


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    records = load_records(Path(args.dataset_path))
    conf_rows = [
        r
        for r in records
        if isinstance(r, dict)
        and is_confidence_match(r.get("confidence"), args.conf_target, args.conf_eps)
    ]
    if args.max_rows is not None and args.max_rows > 0:
        conf_rows = conf_rows[: args.max_rows]

    LOGGER.info(
        "Records: total=%d, confidence==%.6f matched=%d",
        len(records),
        args.conf_target,
        len(conf_rows),
    )

    predictor = MediaFilenameNERPredictor(
        model_dir_or_name=args.model_dir,
        max_length=args.max_length,
        device=args.device,
    )

    per_field_total = {f: 0 for f in FIELDS}
    per_field_match = {f: 0 for f in FIELDS}
    season_raw_total = 0
    season_raw_match = 0
    overall_match = 0
    mismatches: List[Dict[str, Any]] = []

    for idx, row in enumerate(conf_rows):
        raw_path = row.get("raw_path") if isinstance(row.get("raw_path"), str) else ""
        filename = row.get("filename") if isinstance(row.get("filename"), str) else ""
        model_input = compose_path_aware_text(
            raw_path=raw_path,
            filename=filename,
            mode=args.path_aware_mode,
        )
        pred = predictor.predict(model_input, return_spans=False)
        target = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}

        row_ok = True
        field_debug: Dict[str, Dict[str, Any]] = {}
        for field in FIELDS:
            t_norm = normalize_value(field, target.get(field))
            p_norm = normalize_value(field, pred.get(field))
            matched = t_norm == p_norm

            per_field_total[field] += 1
            if matched:
                per_field_match[field] += 1
            else:
                row_ok = False

            info: Dict[str, Any] = {
                "target_raw": target.get(field),
                "pred_raw": pred.get(field),
                "target_norm": t_norm,
                "pred_norm": p_norm,
                "match": matched,
            }

            if field == SEASON_EP_FIELD:
                t_raw_cmp = normalize_value_raw_for_se(target.get(field))
                p_raw_cmp = normalize_value_raw_for_se(pred.get(field))
                raw_match = t_raw_cmp == p_raw_cmp
                season_raw_total += 1
                if raw_match:
                    season_raw_match += 1
                info["target_raw_cmp"] = t_raw_cmp
                info["pred_raw_cmp"] = p_raw_cmp
                info["raw_match"] = raw_match

            field_debug[field] = info

        if row_ok:
            overall_match += 1
        elif len(mismatches) < max(0, int(args.keep_mismatch_samples)):
            mismatches.append(
                {
                    "index": idx,
                    "raw_path": raw_path,
                    "filename": filename,
                    "target": target,
                    "pred": pred,
                    "fields": field_debug,
                }
            )

        if args.log_every > 0 and (idx + 1) % args.log_every == 0:
            LOGGER.info("Progress: %d/%d", idx + 1, len(conf_rows))

    total = len(conf_rows)
    per_field = {}
    for field in FIELDS:
        t = per_field_total[field]
        m = per_field_match[field]
        per_field[field] = {
            "total": t,
            "match": m,
            "accuracy": (m / t) if t else 0.0,
        }

    summary = {
        "dataset_path": args.dataset_path,
        "model_dir": args.model_dir,
        "path_aware_mode": args.path_aware_mode,
        "confidence_target": args.conf_target,
        "confidence_eps": args.conf_eps,
        "total_records": len(records),
        "evaluated_records": total,
        "overall_exact_match": {
            "match": overall_match,
            "total": total,
            "accuracy": (overall_match / total) if total else 0.0,
        },
        "per_field": per_field,
        "season_episode_compare": {
            "raw_match": season_raw_match,
            "raw_total": season_raw_total,
            "raw_accuracy": (season_raw_match / season_raw_total)
            if season_raw_total
            else 0.0,
            "canonical_match": per_field.get(SEASON_EP_FIELD, {}).get("match", 0),
            "canonical_total": per_field.get(SEASON_EP_FIELD, {}).get("total", 0),
            "canonical_accuracy": per_field.get(SEASON_EP_FIELD, {}).get(
                "accuracy", 0.0
            ),
        },
        "kept_mismatch_samples": len(mismatches),
    }
    return {"summary": summary, "mismatches": mismatches}


def main() -> None:
    args = parse_args()
    setup_shared_logging(args.log_level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = evaluate(args)
    summary = results["summary"]
    mismatches = results["mismatches"]

    summary_path = output_dir / "summary.json"
    mismatches_path = output_dir / "mismatch_samples.json"
    legacy_mismatches_path = output_dir / "season_episode_mismatches.json"

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with mismatches_path.open("w", encoding="utf-8") as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=2)
    # Backward-compatible artifact name kept for previous runs/tools.
    with legacy_mismatches_path.open("w", encoding="utf-8") as f:
        json.dump(mismatches, f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved summary: %s", summary_path)
    LOGGER.info("Saved mismatches: %s", mismatches_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
