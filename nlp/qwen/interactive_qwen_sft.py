#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interactive CLI for Qwen LoRA filename parser.

Usage:
  python nlp/qwen/interactive_qwen_sft.py
  python nlp/qwen/interactive_qwen_sft.py --text "Movie.2024.1080p.WEB-DL.H264.AAC-GRP.mkv"
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Iterable, Optional

from nlp.shared import setup_logging as setup_shared_logging
from nlp.qwen.qwen_sft_parser.inference import (
    PATH_AWARE_MODES,
    QwenFilenameJsonParser,
    safe_basename,
)


LOGGER = logging.getLogger("interactive_qwen_sft")
EXIT_WORDS = {"exit", "quit", "q", ":q"}


def setup_logging() -> None:
    setup_shared_logging("INFO")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive Qwen LoRA filename parser."
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default=None,
        help="LoRA adapter directory. If omitted, auto-detect under outputs/.",
    )
    parser.add_argument(
        "--device", type=str, default=None, help="cuda/cpu, default auto"
    )
    parser.add_argument(
        "--path_aware_mode",
        type=str,
        default="raw_plus_basename",
        choices=PATH_AWARE_MODES,
        help="How to compose input text for model prompt.",
    )
    parser.add_argument(
        "--merge_lora", action="store_true", help="Merge LoRA after loading."
    )
    parser.add_argument(
        "--strict_json",
        action="store_true",
        help="Raise error on malformed JSON output.",
    )

    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)

    parser.add_argument(
        "--text", type=str, default=None, help="Single-line mode (non-interactive)."
    )
    parser.add_argument(
        "--show_raw_generation",
        action="store_true",
        help="Include raw model text in output JSON.",
    )
    return parser.parse_args()


def is_valid_adapter_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "adapter_config.json").exists()
    has_weights = (
        any(path.glob("adapter_model*.safetensors"))
        or (path / "adapter_model.bin").exists()
    )
    return has_config and has_weights


def iter_adapter_candidates() -> Iterable[Path]:
    explicit = Path(
        "outputs/qwen_sft_parser_from_pve/qwen3_1p7b_p40_20260402_030432_final/adapter"
    )
    if explicit.exists():
        yield explicit

    for root in [
        Path("outputs/qwen_sft_parser_from_pve"),
        Path("outputs/qwen_sft_parser"),
    ]:
        if not root.exists():
            continue

        # Prefer direct run-level "adapter" directories.
        yield from sorted(
            root.glob("**/adapter"), key=lambda p: p.stat().st_mtime, reverse=True
        )

        # Fallback: allow checkpoint directories containing adapter_model.
        yield from sorted(
            root.glob("**/checkpoint-*"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )


def resolve_adapter_dir(adapter_dir: Optional[str]) -> Path:
    if adapter_dir:
        p = Path(adapter_dir)
        if not is_valid_adapter_dir(p):
            raise FileNotFoundError(
                f"Invalid adapter_dir: {p}. Expect adapter_config.json + adapter_model weights."
            )
        return p

    for candidate in iter_adapter_candidates():
        if is_valid_adapter_dir(candidate):
            return candidate

    raise FileNotFoundError(
        "No valid adapter found. Please provide --adapter_dir explicitly."
    )


def build_output(
    input_text: str,
    result: Dict[str, object],
    show_raw_generation: bool,
) -> Dict[str, object]:
    out: Dict[str, object] = {
        "input": input_text,
        "filename": safe_basename(input_text),
        "parsed": result.get("parsed", {}),
    }
    if "error" in result:
        out["error"] = result.get("error")
    if show_raw_generation:
        out["raw_generation"] = result.get("raw_generation", "")
    return out


def run_one(
    model: QwenFilenameJsonParser,
    text: str,
    args: argparse.Namespace,
) -> Dict[str, object]:
    result = model.parse(
        path_or_name=text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        strict_json=args.strict_json,
    )
    return build_output(text, result, show_raw_generation=args.show_raw_generation)


def interactive_loop(model: QwenFilenameJsonParser, args: argparse.Namespace) -> None:
    print("Qwen SFT Interactive Mode")
    print("Input one line and press Enter. Type exit/quit/q to stop.")
    while True:
        try:
            raw = input("path> ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        text = raw.strip()
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            break

        try:
            out = run_one(model, text, args)
            print(json.dumps(out, ensure_ascii=False, indent=2))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                json.dumps(
                    {"input": text, "error": str(exc)}, ensure_ascii=False, indent=2
                )
            )


def stdin_batch_loop(model: QwenFilenameJsonParser, args: argparse.Namespace) -> None:
    for raw in sys.stdin:
        text = raw.strip()
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            break
        try:
            out = run_one(model, text, args)
            print(json.dumps(out, ensure_ascii=False, indent=2))
        except Exception as exc:  # pylint: disable=broad-except
            print(
                json.dumps(
                    {"input": text, "error": str(exc)}, ensure_ascii=False, indent=2
                )
            )


def main() -> None:
    args = parse_args()
    setup_logging()

    adapter_path = resolve_adapter_dir(args.adapter_dir)
    LOGGER.info("Using adapter_dir: %s", adapter_path)

    model = QwenFilenameJsonParser(
        base_model=args.base_model,
        adapter_dir=str(adapter_path),
        device=args.device,
        merge_lora=args.merge_lora,
        path_aware_mode=args.path_aware_mode,
    )

    if args.text is not None:
        out = run_one(model, args.text, args)
        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    if sys.stdin.isatty():
        interactive_loop(model, args)
    else:
        stdin_batch_loop(model, args)


if __name__ == "__main__":
    main()
