#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run model inference on parsed_dataset_low_confidence.json using raw_path as input,
then export a side-by-side CSV for manual comparison.

CSV layout (interleaved by field):
  orig_title, pred_title, match_title,
  orig_zh_title, pred_zh_title, match_zh_title, ...

Example:
  python nlp/ner/compare_low_confidence_with_model.py
  python nlp/ner/compare_low_confidence_with_model.py --max_rows 200 --output_csv outputs/low_conf_compare.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from nlp.shared import setup_logging as setup_shared_logging
from nlp.ner.train_media_filename_ner import (
    PARSED_KEY_TO_ENTITY,
    MediaFilenameNERPredictor,
)


LOGGER = logging.getLogger("low_conf_compare")

PARSED_FIELDS: List[str] = list(PARSED_KEY_TO_ENTITY.keys())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare low-confidence parsed fields with model predictions."
    )
    parser.add_argument(
        "--input_json", type=str, default="parsed_dataset_low_confidence.json"
    )
    parser.add_argument("--model_dir", type=str, default="outputs/media_filename_ner")
    parser.add_argument(
        "--output_csv", type=str, default="outputs/low_confidence_model_compare.csv"
    )
    parser.add_argument(
        "--input_field",
        type=str,
        default="raw_path",
        choices=["raw_path", "filename"],
        help="Which field from JSON record to feed into the model.",
    )
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument(
        "--device", type=str, default=None, help="cuda/cpu, default auto."
    )
    parser.add_argument(
        "--max_rows", type=int, default=None, help="Optional cap for quick debugging."
    )
    parser.add_argument(
        "--encoding",
        type=str,
        default="utf-8-sig",
        help="CSV encoding; utf-8-sig is Excel-friendly on Windows.",
    )
    parser.add_argument(
        "--with_spans",
        action="store_true",
        help="Append predicted spans as JSON string.",
    )
    parser.add_argument("--log_every", type=int, default=100)
    return parser.parse_args()


def setup_logging() -> None:
    setup_shared_logging("INFO")


def load_records(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Input JSON root must be a list.")
    return data


def to_str_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return str(value)


def normalize_for_match(value: Any) -> str:
    """
    Lightweight normalization for easier manual comparison flags.
    """
    if value is None:
        return ""
    text = str(value).strip().lower()
    # remove spaces only; keep punctuation so we don't over-normalize.
    return text.replace(" ", "")


def build_csv_headers(with_spans: bool) -> List[str]:
    headers: List[str] = [
        "index",
        "confidence",
        "raw_path",
        "filename",
        "model_input_field",
        "model_input_text",
    ]
    for field in PARSED_FIELDS:
        headers.extend([f"orig_{field}", f"pred_{field}", f"match_{field}"])
    if with_spans:
        headers.append("pred_spans_json")
    return headers


def build_row(
    idx: int,
    record: Dict[str, Any],
    model_input_field: str,
    model_input_text: str,
    pred: Dict[str, Optional[str]],
    spans: Optional[List[Dict[str, Any]]],
    with_spans: bool,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "index": idx,
        "confidence": record.get("confidence"),
        "raw_path": record.get("raw_path", ""),
        "filename": record.get("filename", ""),
        "model_input_field": model_input_field,
        "model_input_text": model_input_text,
    }

    parsed_orig = record.get("parsed") or {}
    if not isinstance(parsed_orig, dict):
        parsed_orig = {}

    for field in PARSED_FIELDS:
        orig_val = parsed_orig.get(field)
        pred_val = pred.get(field)
        row[f"orig_{field}"] = to_str_or_empty(orig_val)
        row[f"pred_{field}"] = to_str_or_empty(pred_val)
        row[f"match_{field}"] = int(
            normalize_for_match(orig_val) == normalize_for_match(pred_val)
        )

    if with_spans:
        row["pred_spans_json"] = json.dumps(spans or [], ensure_ascii=False)

    return row


def main() -> None:
    args = parse_args()
    setup_logging()

    input_path = Path(args.input_json)
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(input_path)
    if args.max_rows is not None and args.max_rows > 0:
        records = records[: args.max_rows]
    LOGGER.info("Loaded %d records from %s", len(records), input_path)

    predictor = MediaFilenameNERPredictor(
        model_dir_or_name=args.model_dir,
        max_length=args.max_length,
        device=args.device,
    )

    headers = build_csv_headers(with_spans=args.with_spans)
    processed = 0
    failed = 0

    with output_path.open("w", encoding=args.encoding, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()

        for idx, record in enumerate(records):
            model_input_text = to_str_or_empty(record.get(args.input_field)).strip()
            if not model_input_text:
                failed += 1
                pred: Dict[str, Optional[str]] = {k: None for k in PARSED_FIELDS}
                spans: List[Dict[str, Any]] = []
            else:
                try:
                    if args.with_spans:
                        pred, spans = predictor.predict(
                            model_input_text, return_spans=True
                        )
                    else:
                        pred = predictor.predict(model_input_text, return_spans=False)
                        spans = []
                except Exception as exc:  # pylint: disable=broad-except
                    failed += 1
                    LOGGER.warning("Predict failed at idx=%d: %s", idx, exc)
                    pred = {k: None for k in PARSED_FIELDS}
                    spans = []

            row = build_row(
                idx=idx,
                record=record,
                model_input_field=args.input_field,
                model_input_text=model_input_text,
                pred=pred,
                spans=spans,
                with_spans=args.with_spans,
            )
            writer.writerow(row)
            processed += 1

            if args.log_every > 0 and processed % args.log_every == 0:
                LOGGER.info("Progress: %d/%d", processed, len(records))

    LOGGER.info(
        "Done. processed=%d failed=%d output=%s", processed, failed, output_path
    )


if __name__ == "__main__":
    main()
