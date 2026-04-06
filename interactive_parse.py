"""Interactive media filename parser with selectable backend.

Usage:
  python interactive_parse.py --backend rule
  python interactive_parse.py --backend ner --model_dir outputs/media_filename_ner
  python interactive_parse.py --backend qwen --adapter_dir outputs/qwen_sft_parser/<run>/adapter
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Dict

from media_parser import create_backend_parser


EXIT_WORDS = {"exit", "quit", "q", ":q"}
BACKEND_CHOICES = ("rule", "ner", "qwen")
PATH_AWARE_MODES = ("basename", "raw_path", "raw_plus_basename")
LOGGER = logging.getLogger("interactive_parse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Interactive parser with selectable backend."
    )
    parser.add_argument("--backend", type=str, default="rule", choices=BACKEND_CHOICES)
    parser.add_argument(
        "--text", type=str, default=None, help="Single-line mode (non-interactive)."
    )

    parser.add_argument("--model_dir", type=str, default="outputs/media_filename_ner")
    parser.add_argument("--max_length", type=int, default=192)

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--adapter_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--merge_lora", action="store_true")
    parser.add_argument("--strict_json", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument("--show_raw_generation", action="store_true")
    parser.add_argument(
        "--path_aware_mode",
        type=str,
        default="raw_plus_basename",
        choices=PATH_AWARE_MODES,
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def build_parser(args: argparse.Namespace):
    return create_backend_parser(
        backend=args.backend,
        model_dir=args.model_dir,
        max_length=args.max_length,
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        device=args.device,
        merge_lora=args.merge_lora,
        strict_json=args.strict_json,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        show_raw_generation=args.show_raw_generation,
        path_aware_mode=args.path_aware_mode,
    )


def run_one(parser_obj, text: str) -> Dict[str, object]:
    result = parser_obj.parse(text)
    result["input"] = text
    return result


def interactive_loop(parser_obj) -> None:
    print("Media Parser Interactive Mode")
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
            print(json.dumps(run_one(parser_obj, text), ensure_ascii=False, indent=2))
        except Exception as exc:  # Keep REPL alive if one line fails.
            print(
                json.dumps(
                    {"input": text, "error": str(exc)}, ensure_ascii=False, indent=2
                )
            )


def stdin_batch_loop(parser_obj) -> None:
    for raw in sys.stdin:
        text = raw.strip()
        if not text:
            continue
        if text.lower() in EXIT_WORDS:
            break
        try:
            print(json.dumps(run_one(parser_obj, text), ensure_ascii=False))
        except Exception as exc:
            print(json.dumps({"input": text, "error": str(exc)}, ensure_ascii=False))


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    try:
        parser_obj = build_parser(args)
    except Exception as exc:
        LOGGER.error("Parser init failed: %s", exc)
        raise

    if args.text is not None:
        text = args.text.strip()
        if not text:
            raise ValueError("--text is empty.")
        print(json.dumps(run_one(parser_obj, text), ensure_ascii=False, indent=2))
        return

    if sys.stdin.isatty():
        interactive_loop(parser_obj)
    else:
        stdin_batch_loop(parser_obj)


if __name__ == "__main__":
    main()
