#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Generate an augmented JSON feed for NER hard-negative training.

This script mines episode-confusion rows (e.g. 2024 / 1080 / 1134) and
duplicates them, so downstream training can up-weight these difficult cases.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, List, Sequence

from nlp.shared import PATH_AWARE_MODES, compose_path_aware_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build hard-negative augmented feed for NER."
    )
    parser.add_argument("--input_json", type=str, default="parsed_dataset.json")
    parser.add_argument(
        "--output_json", type=str, default="data/ner_hard_negative_feed.json"
    )
    parser.add_argument(
        "--repeat_factor",
        type=int,
        default=3,
        help="Times to keep each hard-negative row.",
    )
    parser.add_argument(
        "--tokens",
        type=str,
        default="2024,1080,1134",
        help="Comma-separated confusion tokens.",
    )
    parser.add_argument(
        "--path_aware_mode",
        type=str,
        default="raw_plus_basename",
        choices=PATH_AWARE_MODES,
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="append",
        choices=("append", "hard_only"),
        help="append: original + duplicated hard rows; hard_only: only duplicated hard rows.",
    )
    return parser.parse_args()


def is_hard_negative_candidate(
    text: str,
    parsed: Dict[str, Any],
    tokens: Sequence[str],
) -> bool:
    season_episode = parsed.get("season_episode")
    if season_episode is None or not str(season_episode).strip():
        return False

    lowered = text.lower()
    token_hit = any(tok and str(tok).strip().lower() in lowered for tok in tokens)
    has_bracket_number = re.search(r"\[(\d{3,4})\]", text) is not None
    has_year_like = re.search(r"(?<!\d)(19\d{2}|20[0-3]\d)(?!\d)", text) is not None
    has_res_like = (
        re.search(
            r"(?i)(?<!\d)(360|480|540|576|720|900|1080|1440|2160|4320)(?!\d)",
            text,
        )
        is not None
    )
    has_episode_like = (
        re.search(
            r"(?i)\bS\d{1,3}E\d{1,4}\b|\b\d{1,3}x\d{1,4}\b|\[\d{2,4}\]|第.+[季集话話]",
            text,
        )
        is not None
    )
    return bool(
        has_episode_like
        and (token_hit or (has_bracket_number and has_year_like and has_res_like))
    )


def build_path_aware_text(row: Dict[str, Any], mode: str) -> str:
    return compose_path_aware_text(
        raw_path=str(row.get("raw_path") or ""),
        filename=str(row.get("filename") or ""),
        mode=mode,
    )


def main() -> None:
    args = parse_args()
    input_path = Path(args.input_json)
    output_path = Path(args.output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(input_path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError("Input JSON root must be list.")

    tokens = [tok.strip() for tok in str(args.tokens).split(",") if tok.strip()]
    repeat_factor = max(1, int(args.repeat_factor))

    hard_rows: List[Dict[str, Any]] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        parsed = row.get("parsed") if isinstance(row.get("parsed"), dict) else {}
        text = build_path_aware_text(row, mode=args.path_aware_mode)
        if not text:
            continue
        if not is_hard_negative_candidate(text, parsed, tokens):
            continue
        hard_rows.append(row)

    duplicated: List[Dict[str, Any]] = []
    for row in hard_rows:
        for rep in range(repeat_factor):
            row_copy = dict(row)
            row_copy["_hard_negative"] = True
            row_copy["_hard_negative_round"] = rep + 1
            duplicated.append(row_copy)

    if args.mode == "hard_only":
        out_rows = duplicated
    else:
        out_rows = list(rows) + duplicated

    output_path.write_text(
        json.dumps(out_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "input_records": len(rows),
                "hard_negative_candidates": len(hard_rows),
                "repeat_factor": repeat_factor,
                "generated_rows": len(duplicated),
                "output_records": len(out_rows),
                "output_json": str(output_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
