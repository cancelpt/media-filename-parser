# media-filename-parser

[![CI](https://github.com/cancelpt/media-filename-parser/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cancelpt/media-filename-parser/actions/workflows/ci.yml)
[![Deploy Pages](https://github.com/cancelpt/media-filename-parser/actions/workflows/deploy-pages.yml/badge.svg?branch=main)](https://github.com/cancelpt/media-filename-parser/actions/workflows/deploy-pages.yml)
[![Demo](https://img.shields.io/badge/demo-online-0f766e?logo=githubpages&logoColor=white)](https://cancelpt.github.io/media-filename-parser/)

[简体中文](README.zh-CN.md)

A Python-first media filename parser with a reusable rule-based package, optional NLP research pipelines, and a web demo.

## Highlights

- Installable package: `media_filename_parser` (rules only)
- CLI entrypoints for single parse and batch parse
- Unified local script for `rule` / `ner` / `qwen` backends
- Browser demo in `web/app/`

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## Rule Package Usage

```python
from media_filename_parser import RuleParser, parse_filename
import json

single = parse_filename("[CancelWEB][CONAN][1134][1080P].mkv")
print("single:")
print(json.dumps(single, ensure_ascii=False, indent=2))

engine = RuleParser()
batch = engine.parse_many(
    [
        "Stranger.S02E14.1080p.BluRay.x265.10bit-CancelHD.mkv",
        "[CancelSub] Cardfight Vanguard overDress S03E13 [1080P][WEB-DL][AAC AVC].mp4",
    ]
)
print(f"batch size: {len(batch)}")
print("batch[0]:")
print(json.dumps(batch[0], ensure_ascii=False, indent=2))
```

For application integration (for example `torrent-transfer`), prefer typed payload and
query-name helper:

```python
from media_filename_parser import build_query_name, parse_filename_typed

typed = parse_filename_typed("Black.Mirror.S02E03.2011.mkv")
print(typed.title, typed.year, typed.season_episode)

query_name = build_query_name("Black.Mirror.S02E03.2011.mkv")
print(query_name)  # Black Mirror.2011.S02
```

Sample output (excerpt):

```json
{
  "raw_path": "[CancelWEB][CONAN][1134][1080P].mkv",
  "filename": "[CancelWEB][CONAN][1134][1080P].mkv",
  "parsed": {
    "title": "CONAN",
    "season_episode": "1134",
    "resolution": "1080P",
    "group": "CancelWEB"
  },
  "confidence": 0.9
}
```

CLI:

```bash
python -m media_filename_parser parse --text "[CancelWEB][CONAN][1134][1080P].mkv" --pretty
python -m media_filename_parser batch --input_file scrape_files_list_merged.txt --output_file parsed_dataset.json
# or
media-filename-parser parse --text "[CancelWEB][CONAN][1134][1080P].mkv" --pretty
```

`batch` writes results to `parsed_dataset.json` as a JSON array.

## Unified Backend Script

```bash
python media_parser.py --backend rule --text "[CancelWEB][CONAN][1134][1080P].mkv"
python media_parser.py --backend ner --model_dir outputs/media_filename_ner --text "..."
python media_parser.py --backend qwen --adapter_dir outputs/qwen_sft_parser/<run>/adapter --text "..."
```

NLP training/evaluation details are maintained in [nlp/README.md](nlp/README.md).

## Web Demo

Live demo: [https://cancelpt.github.io/media-filename-parser/](https://cancelpt.github.io/media-filename-parser/)

Web demo details and local commands are maintained in [web/app/README.md](web/app/README.md).

Typical flow from repository root:

```bash
python -m tools.export_parser_constants
python -m tools.check_web_parser_parity --input scrape_files_list_merged.txt --limit 200
npm --prefix web/app install
npm --prefix web/app run test
npm --prefix web/app run build-dist
python -m http.server 8080 --directory web/app/dist
```

## Repository Layout

- `src/media_filename_parser/`: installable rule parser package
- `media_parser.py`: unified parser entrypoint (`rule` / `ner` / `qwen`)
- `interactive_parse.py`: interactive parser CLI
- `nlp/`: NLP training/evaluation/inference pipelines (see `nlp/README.md`)
- `web/app/`: GitHub Pages-ready web parser app (see `web/app/README.md`)
- `tools/export_parser_constants.py`: export parser constants to web
- `tools/check_web_parser_parity.py`: parity checks between Python and web parser
- `tests/`: smoke/regression tests

## CI/CD

- CI workflow: `.github/workflows/ci.yml`
  - Ruff + Pylint + Pytest
  - Python package build and smoke checks
  - Web build artifact verification
- GitHub Pages workflow: `.github/workflows/deploy-pages.yml`
  - Build and deploy `web/app/dist`

## Notes

- Large datasets, model checkpoints, and generated outputs are ignored by `.gitignore`.
- Keep private tokens (for example `HF_TOKEN`) outside the repository.
