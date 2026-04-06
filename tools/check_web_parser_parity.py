"""Check parity between Python rule parser and web TS parser.

Usage:
  python tools/check_web_parser_parity.py --input scrape_files_list_merged.txt
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO_ROOT = Path(__file__).resolve().parent.parent
try:
    from media_filename_parser.rules.parser import parse_filename as parse_rule
except ModuleNotFoundError:
    SRC_DIR = REPO_ROOT / "src"
    if SRC_DIR.exists():
        sys.path.insert(0, str(SRC_DIR))
    from media_filename_parser.rules.parser import parse_filename as parse_rule


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare Python parser and web JS parser output parity."
    )
    parser.add_argument("--input", type=str, default="scrape_files_list_merged.txt")
    parser.add_argument("--limit", type=int, default=0, help="0 means full file.")
    parser.add_argument("--show_examples", type=int, default=5)
    return parser.parse_args()


def _normalize(value):
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped if stripped else None
    return value


def load_lines(path: Path, limit: int) -> List[str]:
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    lines = [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit > 0:
        lines = lines[:limit]
    return lines


def run_js_batch(lines: List[str], repo_root: Path) -> List[Dict]:
    payload = "\n".join(lines) + "\n"
    if os.name == "nt":
        cmd = ["cmd", "/c", "npx --yes tsx web/app/src/parser/cli.ts"]
    else:
        cmd = ["npx", "--yes", "tsx", "web/app/src/parser/cli.ts"]
    proc = subprocess.Popen(
        cmd,
        cwd=str(repo_root),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    stdout, stderr = proc.communicate(payload)
    if proc.returncode != 0:
        raise RuntimeError(f"Node parser execution failed:\n{stderr}")

    rows = [json.loads(line) for line in stdout.splitlines() if line.strip()]
    if len(rows) != len(lines):
        raise RuntimeError(
            f"JS output line count mismatch: expected={len(lines)} got={len(rows)}."
        )
    return rows


def compare_one(py_res: Dict, js_res: Dict) -> Dict[str, Tuple[object, object]]:
    diffs: Dict[str, Tuple[object, object]] = {}

    py_parsed = py_res.get("parsed", {})
    js_parsed = js_res.get("parsed", {})
    for field in FIELDS:
        left = _normalize(py_parsed.get(field))
        right = _normalize(js_parsed.get(field))
        if left != right:
            diffs[field] = (left, right)

    py_conf = float(py_res.get("confidence", 0.0))
    js_conf = float(js_res.get("confidence", 0.0))
    if abs(py_conf - js_conf) > 1e-9:
        diffs["confidence"] = (py_conf, js_conf)

    return diffs


def main() -> int:
    args = parse_args()
    repo_root = REPO_ROOT
    input_path = (repo_root / args.input).resolve()
    lines = load_lines(input_path, args.limit)
    js_rows = run_js_batch(lines, repo_root=repo_root)

    mismatch_count = 0
    examples = []
    for idx, (line, js_res) in enumerate(zip(lines, js_rows)):
        py_res = parse_rule(line)
        diffs = compare_one(py_res, js_res)
        if diffs:
            mismatch_count += 1
            if len(examples) < args.show_examples:
                examples.append((idx, line, diffs))

    print(f"Checked: {len(lines)}")
    print(f"Mismatched lines: {mismatch_count}")

    if examples:
        print("\nExamples:")
        for idx, line, diffs in examples:
            print(f"- index={idx}")
            print(f"  path={line}")
            for key, (left, right) in diffs.items():
                print(f"  {key}: py={left!r} js={right!r}")

    return 1 if mismatch_count else 0


if __name__ == "__main__":
    sys.exit(main())
