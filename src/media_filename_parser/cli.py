"""CLI for the installable rule-based parser package."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable, List

from .parser import RuleParser


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Rule-based media filename parser (NLP-free package entrypoint)."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    parse_parser = subparsers.add_parser("parse", help="Parse a single path/filename.")
    parse_parser.add_argument("--text", type=str, required=True, help="Input text.")
    parse_parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print JSON output."
    )

    batch_parser = subparsers.add_parser(
        "batch", help="Parse line-based input and write JSON array output."
    )
    batch_parser.add_argument(
        "--input_file",
        type=Path,
        required=True,
        help="Input txt file; one filename/path per line.",
    )
    batch_parser.add_argument(
        "--output_file",
        type=Path,
        required=True,
        help="Output JSON path.",
    )
    batch_parser.add_argument(
        "--low_conf_file",
        type=Path,
        default=None,
        help="Optional output for records with confidence<1.0.",
    )
    batch_parser.add_argument("--encoding", type=str, default="utf-8")
    return parser


def _load_non_empty_lines(path: Path, encoding: str) -> List[str]:
    """Load non-empty lines from a text file."""
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    with path.open("r", encoding=encoding) as f:
        return [line.strip() for line in f if line.strip()]


def _write_json(path: Path, payload: object) -> None:
    """Write JSON payload to disk using UTF-8."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _iter_low_conf(results: Iterable[dict]) -> List[dict]:
    """Collect and sort records with confidence lower than 1.0."""
    low_conf = [
        item
        for item in results
        if isinstance(item.get("confidence"), (int, float))
        and float(item["confidence"]) < 1.0
    ]
    low_conf.sort(key=lambda x: x.get("confidence", 0.0))
    return low_conf


def main(argv: list[str] | None = None) -> int:
    """Run package CLI and return process exit code."""
    parser = build_parser()
    args = parser.parse_args(argv)
    engine = RuleParser()

    if args.command == "parse":
        text = args.text.strip()
        if not text:
            raise ValueError("--text is empty.")
        result = engine.parse(text)
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
        return 0

    if args.command == "batch":
        lines = _load_non_empty_lines(args.input_file, encoding=args.encoding)
        results = engine.parse_many(lines)
        _write_json(args.output_file, results)
        print(f"wrote {len(results)} result(s) to {args.output_file}")

        if args.low_conf_file is not None:
            low_conf = _iter_low_conf(results)
            _write_json(args.low_conf_file, low_conf)
            print(
                f"wrote {len(low_conf)} low-confidence result(s) "
                f"to {args.low_conf_file}"
            )
        return 0

    parser.print_help(sys.stderr)
    return 2
