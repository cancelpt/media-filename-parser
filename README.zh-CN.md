# media-filename-parser

[![CI](https://github.com/cancelpt/media-filename-parser/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/cancelpt/media-filename-parser/actions/workflows/ci.yml)
[![Deploy Pages](https://github.com/cancelpt/media-filename-parser/actions/workflows/deploy-pages.yml/badge.svg?branch=main)](https://github.com/cancelpt/media-filename-parser/actions/workflows/deploy-pages.yml)
[![Demo](https://img.shields.io/badge/demo-online-0f766e?logo=githubpages&logoColor=white)](https://cancelpt.github.io/media-filename-parser/)

[English](README.md)

一个以 Python 为核心的媒体文件名解析项目，包含可复用的规则解析包、可选的 NLP 训练流水线，以及 Web 演示页面。

## 项目亮点

- 可安装包：`media_filename_parser`（仅规则解析能力）
- 提供单条解析和批量解析的 CLI
- 提供统一脚本，支持 `rule` / `ner` / `qwen` 后端
- 提供浏览器端演示，位于 `web/app/`

## 快速开始

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -U pip
pip install -r requirements.txt
pip install -e .
```

## 规则解析包用法

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

示例输出（节选）：

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

CLI：

```bash
python -m media_filename_parser parse --text "[CancelWEB][CONAN][1134][1080P].mkv" --pretty
python -m media_filename_parser batch --input_file scrape_files_list_merged.txt --output_file parsed_dataset.json
# 或
media-filename-parser parse --text "[CancelWEB][CONAN][1134][1080P].mkv" --pretty
```

`batch` 命令会将结果写入 `parsed_dataset.json`（JSON 数组）。

## 统一后端脚本

```bash
python media_parser.py --backend rule --text "[CancelWEB][CONAN][1134][1080P].mkv"
python media_parser.py --backend ner --model_dir outputs/media_filename_ner --text "..."
python media_parser.py --backend qwen --adapter_dir outputs/qwen_sft_parser/<run>/adapter --text "..."
```

NLP 的训练与评估细节放在 [nlp/README.zh-CN.md](nlp/README.zh-CN.md)。

## Web 演示

在线 Demo：[https://cancelpt.github.io/media-filename-parser/](https://cancelpt.github.io/media-filename-parser/)

Web 端说明和本地运行命令见 [web/app/README.zh-CN.md](web/app/README.zh-CN.md)。

仓库根目录下的典型流程：

```bash
python -m tools.export_parser_constants
python -m tools.check_web_parser_parity --input scrape_files_list_merged.txt --limit 200
npm --prefix web/app install
npm --prefix web/app run test
npm --prefix web/app run build-dist
python -m http.server 8080 --directory web/app/dist
```

## 仓库结构

- `src/media_filename_parser/`：可安装的规则解析包
- `media_parser.py`：统一解析入口（`rule` / `ner` / `qwen`）
- `interactive_parse.py`：交互式解析 CLI
- `nlp/`：NLP 训练/评估/推理流水线（见 `nlp/README.zh-CN.md`）
- `web/app/`：GitHub Pages 演示应用（见 `web/app/README.zh-CN.md`）
- `tools/export_parser_constants.py`：导出规则常量给 Web 端
- `tools/check_web_parser_parity.py`：Python 与 Web 解析结果一致性检查
- `tests/`：冒烟与回归测试

## CI/CD

- CI 工作流：`.github/workflows/ci.yml`
  - Ruff + Pylint + Pytest
  - Python 包构建与冒烟检查
  - Web 构建产物校验
- GitHub Pages 工作流：`.github/workflows/deploy-pages.yml`
  - 构建并部署 `web/app/dist`

## 数据与安全

- 大型数据集、模型 checkpoint 与生成产物已通过 `.gitignore` 排除。
- 私密令牌（例如 `HF_TOKEN`）应通过环境变量管理，且不应提交到仓库。
