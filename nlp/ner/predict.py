#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Inference entrypoint for media filename NER.

Examples:
  python nlp/ner/predict.py --text "[CancelSub] 卡片戰鬥!! 先導者 overdress 第三季 - 13 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4"
  python nlp/ner/predict.py --input_file sample_filenames.txt --output_file preds.jsonl
  python nlp/ner/predict.py --input_file parsed_dataset.json --with_spans --json_array --pretty
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from nlp.shared import (
    PATH_AWARE_MODES,
    compose_path_aware_text,
    setup_logging as setup_shared_logging,
)
from nlp.ner.train_media_filename_ner import MediaFilenameNERPredictor


LOGGER = logging.getLogger("media_filename_predict")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict structured fields from media filenames."
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        default="outputs/media_filename_ner",
        help="Path to fine-tuned model directory.",
    )

    io_group = parser.add_mutually_exclusive_group()
    io_group.add_argument(
        "--text", type=str, default=None, help="Single filename to parse."
    )
    io_group.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Input file (.txt/.json). If omitted, reads stdin when piped.",
    )

    parser.add_argument(
        "--output_file", type=str, default=None, help="Output path (.json/.jsonl)."
    )
    parser.add_argument(
        "--with_spans", action="store_true", help="Include debug entity spans."
    )
    parser.add_argument(
        "--json_array", action="store_true", help="Write outputs as one JSON array."
    )
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty print JSON output."
    )
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument(
        "--device", type=str, default=None, help="cuda/cpu; default auto."
    )
    parser.add_argument(
        "--path_aware_mode",
        type=str,
        default="raw_plus_basename",
        choices=PATH_AWARE_MODES,
        help="Input text composition strategy for NER inference.",
    )
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    setup_shared_logging(level_name)


def _extract_filenames_from_json(data: Any, path_aware_mode: str) -> List[str]:
    filenames: List[str] = []

    def append_if_valid(value: Any) -> None:
        if isinstance(value, str):
            text = value.strip()
            if text:
                filenames.append(text)

    if isinstance(data, list):
        for item in data:
            if isinstance(item, str):
                append_if_valid(
                    compose_path_aware_text(
                        raw_path=item,
                        filename=item,
                        mode=path_aware_mode,
                    )
                )
                continue
            if isinstance(item, dict):
                raw_path = (
                    item.get("raw_path")
                    if isinstance(item.get("raw_path"), str)
                    else ""
                )
                filename = (
                    item.get("filename")
                    if isinstance(item.get("filename"), str)
                    else ""
                )
                composed = compose_path_aware_text(
                    raw_path=raw_path,
                    filename=filename,
                    mode=path_aware_mode,
                )
                append_if_valid(composed)
    elif isinstance(data, dict):
        for key in ("items", "records", "data"):
            maybe_list = data.get(key)
            if isinstance(maybe_list, list):
                filenames.extend(
                    _extract_filenames_from_json(
                        maybe_list, path_aware_mode=path_aware_mode
                    )
                )
                break

    # Deduplicate but keep order.
    seen = set()
    deduped: List[str] = []
    for name in filenames:
        if name in seen:
            continue
        seen.add(name)
        deduped.append(name)
    return deduped


def load_filenames_from_file(
    path: Path, encoding: str, path_aware_mode: str
) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".json":
        with path.open("r", encoding=encoding) as f:
            data = json.load(f)
        names = _extract_filenames_from_json(data, path_aware_mode=path_aware_mode)
    else:
        with path.open("r", encoding=encoding) as f:
            names = [
                compose_path_aware_text(
                    raw_path=line.strip(),
                    filename=line.strip(),
                    mode=path_aware_mode,
                )
                for line in f
                if line.strip()
            ]

    if not names:
        raise ValueError(f"No usable filenames found in {path}")
    return names


def load_filenames(args: argparse.Namespace) -> List[str]:
    if args.text is not None:
        value = args.text.strip()
        if not value:
            raise ValueError("--text is empty.")
        return [
            compose_path_aware_text(
                raw_path=value,
                filename=value,
                mode=args.path_aware_mode,
            )
        ]

    if args.input_file is not None:
        return load_filenames_from_file(
            Path(args.input_file),
            args.encoding,
            path_aware_mode=args.path_aware_mode,
        )

    if not sys.stdin.isatty():
        names = [
            compose_path_aware_text(
                raw_path=line.strip(),
                filename=line.strip(),
                mode=args.path_aware_mode,
            )
            for line in sys.stdin
            if line.strip()
        ]
        if names:
            return names

    raise ValueError(
        "No input provided. Use --text, --input_file, or pipe lines via stdin."
    )


def run_predictions(
    predictor: MediaFilenameNERPredictor,
    filenames: Iterable[str],
    with_spans: bool,
) -> List[Dict[str, Any]]:
    outputs: List[Dict[str, Any]] = []
    for filename in filenames:
        try:
            if with_spans:
                parsed, spans = predictor.predict(filename, return_spans=True)
                outputs.append({"filename": filename, "parsed": parsed, "spans": spans})
            else:
                parsed = predictor.predict(filename, return_spans=False)
                outputs.append({"filename": filename, "parsed": parsed})
        except Exception as exc:  # pylint: disable=broad-except
            outputs.append({"filename": filename, "error": str(exc)})
    return outputs


def dump_outputs(outputs: List[Dict[str, Any]], args: argparse.Namespace) -> None:
    if args.output_file:
        out_path = Path(args.output_file)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding=args.encoding) as f:
            if args.json_array or out_path.suffix.lower() == ".json":
                indent = 2 if args.pretty else None
                json.dump(outputs, f, ensure_ascii=False, indent=indent)
                f.write("\n")
            else:
                for row in outputs:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
        LOGGER.info("Wrote %d predictions to %s", len(outputs), out_path)
        return

    if len(outputs) == 1:
        print(json.dumps(outputs[0], ensure_ascii=False, indent=2))
        return

    if args.json_array:
        indent = 2 if args.pretty else None
        print(json.dumps(outputs, ensure_ascii=False, indent=indent))
        return

    for row in outputs:
        print(json.dumps(row, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    filenames = load_filenames(args)
    LOGGER.info("Loaded %d filename(s).", len(filenames))

    predictor = MediaFilenameNERPredictor(
        model_dir_or_name=str(model_dir),
        max_length=args.max_length,
        device=args.device,
    )
    outputs = run_predictions(predictor, filenames, with_spans=args.with_spans)
    dump_outputs(outputs, args)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
        sys.exit(130)
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.error("Prediction failed: %s", exc)
        sys.exit(1)
