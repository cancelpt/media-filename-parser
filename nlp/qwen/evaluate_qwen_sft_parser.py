#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate Qwen LoRA Text-to-JSON parser against parsed_dataset.json-style labels.

Features:
1. Load LoRA adapter + base model with qwen_sft_parser.inference
2. Run prediction sample-by-sample
3. Compute exact-match metrics (overall + per-field)
4. Export metrics JSON and detailed CSV for manual review

Usage:
  python nlp/qwen/evaluate_qwen_sft_parser.py --adapter_dir <path_to_adapter>
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from nlp.shared import setup_logging as setup_shared_logging
from nlp.qwen.qwen_sft_parser.inference import (
    FIELDS,
    QwenFilenameJsonParser,
    normalize_frame_rate,
    normalize_season_episode,
)


LOGGER = logging.getLogger("qwen_sft_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Qwen LoRA filename parser on JSON dataset."
    )
    parser.add_argument(
        "--adapter_dir", type=str, required=True, help="Local LoRA adapter directory."
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dataset_json", type=str, default="parsed_dataset.json")
    parser.add_argument(
        "--input_field", type=str, default="raw_path", choices=["raw_path", "filename"]
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda/cpu, default auto"
    )
    parser.add_argument(
        "--merge_lora", action="store_true", help="Merge LoRA before generation."
    )
    parser.add_argument(
        "--strict_json", action="store_true", help="Raise on JSON parse failure."
    )
    parser.add_argument(
        "--min_confidence",
        type=float,
        default=0.8,
        help="Filter records by parser confidence.",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=200,
        help="Limit samples for quick evaluation.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--output_dir", type=str, default="outputs/qwen_sft_parser_eval"
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="Optional custom run suffix."
    )
    parser.add_argument("--log_every", type=int, default=10)
    return parser.parse_args()


def setup_logging() -> None:
    setup_shared_logging("INFO")


def make_run_dir(output_root: Path, run_name: Optional[str]) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    suffix = run_name or datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"eval_{suffix}"
    idx = 2
    while run_dir.exists():
        run_dir = output_root / f"eval_{suffix}_v{idx}"
        idx += 1
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def normalize_scalar(field: str, value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "null":
        return None
    if field == "season_episode":
        return normalize_season_episode(text)
    if field == "frame_rate":
        return normalize_frame_rate(text)
    return text


def load_records(dataset_json: Path, min_confidence: float) -> List[Dict[str, Any]]:
    if not dataset_json.exists():
        raise FileNotFoundError(f"dataset_json not found: {dataset_json}")
    with dataset_json.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("dataset_json root must be a list.")

    out: List[Dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        conf = float(rec.get("confidence", 0.0) or 0.0)
        if conf < min_confidence:
            continue
        parsed = rec.get("parsed")
        if not isinstance(parsed, dict):
            continue
        out.append(rec)
    return out


def select_records(
    records: List[Dict[str, Any]], max_samples: Optional[int], seed: int
) -> List[Dict[str, Any]]:
    if max_samples is None or max_samples <= 0 or len(records) <= max_samples:
        return records
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)
    picked = indices[:max_samples]
    return [records[i] for i in picked]


def evaluate(
    parser: QwenFilenameJsonParser,
    records: List[Dict[str, Any]],
    input_field: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    repetition_penalty: float,
    strict_json: bool,
    log_every: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    field_correct = {field: 0 for field in FIELDS}
    field_total = {field: 0 for field in FIELDS}
    non_null_correct = {field: 0 for field in FIELDS}
    non_null_total = {field: 0 for field in FIELDS}

    object_exact = 0
    parsed_json_fail = 0
    rows: List[Dict[str, Any]] = []

    for idx, rec in enumerate(records):
        model_input = str(rec.get(input_field) or "").strip()
        gt = rec.get("parsed") if isinstance(rec.get("parsed"), dict) else {}

        result = parser.parse(
            path_or_name=model_input,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            strict_json=strict_json,
        )
        pred = result.get("parsed") if isinstance(result.get("parsed"), dict) else {}
        error_flag = result.get("error")
        if error_flag:
            parsed_json_fail += 1

        all_match = True
        row: Dict[str, Any] = {
            "index": idx,
            "raw_path": rec.get("raw_path", ""),
            "filename": rec.get("filename", ""),
            "model_input": model_input,
            "confidence": rec.get("confidence"),
            "json_parse_error": error_flag or "",
        }

        for field in FIELDS:
            gt_v = normalize_scalar(field, gt.get(field))
            pd_v = normalize_scalar(field, pred.get(field))

            match = int(gt_v == pd_v)
            field_correct[field] += match
            field_total[field] += 1
            if gt_v is not None:
                non_null_total[field] += 1
                non_null_correct[field] += match

            row[f"gt_{field}"] = gt_v or ""
            row[f"pred_{field}"] = pd_v or ""
            row[f"match_{field}"] = match

            if match == 0:
                all_match = False

        row["object_exact"] = int(all_match)
        if all_match:
            object_exact += 1
        rows.append(row)

        if log_every > 0 and (idx + 1) % log_every == 0:
            LOGGER.info("Progress: %d/%d", idx + 1, len(records))

    total = len(records)
    summary = {
        "sample_count": total,
        "object_exact_acc": (object_exact / total) if total else 0.0,
        "json_parse_fail_rate": (parsed_json_fail / total) if total else 0.0,
        "field_exact_acc": {
            field: (field_correct[field] / field_total[field])
            if field_total[field]
            else 0.0
            for field in FIELDS
        },
        "field_non_null_acc": {
            field: (non_null_correct[field] / non_null_total[field])
            if non_null_total[field]
            else None
            for field in FIELDS
        },
        "counts": {
            "object_exact": object_exact,
            "json_parse_fail": parsed_json_fail,
            "field_correct": field_correct,
            "field_total": field_total,
            "field_non_null_correct": non_null_correct,
            "field_non_null_total": non_null_total,
        },
    }
    return summary, rows


def write_outputs(
    run_dir: Path,
    args: argparse.Namespace,
    metrics: Dict[str, Any],
    rows: List[Dict[str, Any]],
) -> None:
    metrics_path = run_dir / "metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        payload = {
            "base_model": args.base_model,
            "adapter_dir": args.adapter_dir,
            "dataset_json": args.dataset_json,
            "input_field": args.input_field,
            "min_confidence": args.min_confidence,
            "max_samples": args.max_samples,
            "generation": {
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
                "top_p": args.top_p,
                "repetition_penalty": args.repetition_penalty,
            },
            "metrics": metrics,
        }
        json.dump(payload, f, ensure_ascii=False, indent=2)

    csv_path = run_dir / "detailed_results.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with csv_path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    mismatches = [r for r in rows if r.get("object_exact", 0) == 0]
    mismatch_path = run_dir / "top_mismatches.json"
    with mismatch_path.open("w", encoding="utf-8") as f:
        json.dump(mismatches[:100], f, ensure_ascii=False, indent=2)

    LOGGER.info("Saved metrics: %s", metrics_path)
    LOGGER.info("Saved details: %s", csv_path)
    LOGGER.info("Saved top mismatches: %s", mismatch_path)


def verify_adapter_dir(adapter_dir: Path) -> None:
    if not adapter_dir.exists():
        raise FileNotFoundError(f"adapter_dir not found: {adapter_dir}")
    has_config = (adapter_dir / "adapter_config.json").exists()
    has_weights = (
        any(adapter_dir.glob("adapter_model*.safetensors"))
        or (adapter_dir / "adapter_model.bin").exists()
    )
    if not (has_config and has_weights):
        raise RuntimeError(
            f"adapter_dir looks incomplete: {adapter_dir}. "
            "Need adapter_config.json + adapter_model weights."
        )


def main() -> None:
    args = parse_args()
    setup_logging()

    adapter_dir = Path(args.adapter_dir)
    verify_adapter_dir(adapter_dir)

    records = load_records(Path(args.dataset_json), min_confidence=args.min_confidence)
    records = select_records(records, max_samples=args.max_samples, seed=args.seed)
    if not records:
        raise RuntimeError("No records available after filtering.")

    run_dir = make_run_dir(Path(args.output_dir), args.run_name)
    LOGGER.info("Run dir: %s", run_dir)
    LOGGER.info("Eval samples: %d", len(records))

    parser = QwenFilenameJsonParser(
        base_model=args.base_model,
        adapter_dir=str(adapter_dir),
        device=args.device,
        merge_lora=args.merge_lora,
    )

    metrics, rows = evaluate(
        parser=parser,
        records=records,
        input_field=args.input_field,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        strict_json=args.strict_json,
        log_every=args.log_every,
    )
    write_outputs(run_dir, args, metrics, rows)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
