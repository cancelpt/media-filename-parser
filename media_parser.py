"""Unified media filename parser entrypoint.

This module re-exports core helpers from `media_filename_parser.rules` and
adds selectable backends:
1. `rule` (rule regex parser in media_filename_parser.rules)
2. `ner`  (BIO token-classification model)
3. `qwen` (Qwen LoRA text-to-JSON model)
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from media_filename_parser.rules import (
        DISC_LAYOUT_DIRS,
        GENERIC_SHORT_TITLES,
        P_ACODEC,
        P_FPS,
        P_EP_BRACKET,
        P_MISC,
        P_RES,
        P_SE,
        P_SEP,
        P_SOURCE,
        P_VCODEC,
        P_VHDR,
        P_YEAR,
        calculate_confidence,
        extract_season_episode,
        extract_sports_event_title,
        extract_with_pattern,
        looks_like_technical_disc_title,
        make_pattern,
        parse_filename as _parse_filename_rule,
        resolve_metadata_parent_dir,
    )
except ModuleNotFoundError:
    _REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parent
    _SRC_DIR = _REPO_ROOT_FOR_IMPORT / "src"
    if _SRC_DIR.exists():
        sys.path.insert(0, str(_SRC_DIR))
    from media_filename_parser.rules import (
        DISC_LAYOUT_DIRS,
        GENERIC_SHORT_TITLES,
        P_ACODEC,
        P_FPS,
        P_EP_BRACKET,
        P_MISC,
        P_RES,
        P_SE,
        P_SEP,
        P_SOURCE,
        P_VCODEC,
        P_VHDR,
        P_YEAR,
        calculate_confidence,
        extract_season_episode,
        extract_sports_event_title,
        extract_with_pattern,
        looks_like_technical_disc_title,
        make_pattern,
        parse_filename as _parse_filename_rule,
        resolve_metadata_parent_dir,
    )


parse_filename = _parse_filename_rule

_REPO_ROOT = Path(__file__).resolve().parent
_RE_PATH_SPLIT = re.compile(r"[\\/]+")
_BACKEND_CHOICES = ("rule", "ner", "qwen")
_PATH_AWARE_MODES = ("basename", "raw_path", "raw_plus_basename")
LOGGER = logging.getLogger("media_parser")


__all__ = [
    "calculate_confidence",
    "extract_with_pattern",
    "make_pattern",
    "P_SOURCE",
    "P_RES",
    "P_VCODEC",
    "P_ACODEC",
    "P_FPS",
    "P_VHDR",
    "P_YEAR",
    "P_SE",
    "P_EP_BRACKET",
    "P_SEP",
    "P_MISC",
    "DISC_LAYOUT_DIRS",
    "GENERIC_SHORT_TITLES",
    "extract_season_episode",
    "resolve_metadata_parent_dir",
    "looks_like_technical_disc_title",
    "extract_sports_event_title",
    "parse_filename",
    "parse_with_backend",
    "create_backend_parser",
    "main",
]


def _safe_basename(path_or_name: str) -> str:
    text = (path_or_name or "").strip()
    if not text:
        return ""
    parts = _RE_PATH_SPLIT.split(text)
    return parts[-1] if parts else text


def _compose_path_aware_text(path_or_name: str, mode: str = "raw_plus_basename") -> str:
    text = (path_or_name or "").strip()
    mode = (mode or "raw_plus_basename").strip().lower()
    if mode not in _PATH_AWARE_MODES:
        mode = "raw_plus_basename"
    if not text:
        return ""
    basename = _safe_basename(text)
    if mode == "basename":
        return basename
    if mode == "raw_path":
        return text
    return f"{text} ||| {basename}"


def _clean_rule_result(result: Dict[str, Any]) -> Dict[str, Any]:
    parsed = result.get("parsed", {})
    if isinstance(parsed, dict) and "_inherited_title" in parsed:
        parsed.pop("_inherited_title", None)
    return result


def _is_valid_ner_model_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "config.json").exists()
    has_weights = (
        (path / "model.safetensors").exists()
        or (path / "pytorch_model.bin").exists()
        or (path / "pytorch_model.bin.index.json").exists()
    )
    return has_config and has_weights


def _is_valid_qwen_adapter_dir(path: Path) -> bool:
    if not path.exists() or not path.is_dir():
        return False
    has_config = (path / "adapter_config.json").exists()
    has_weights = (
        any(path.glob("adapter_model*.safetensors"))
        or (path / "adapter_model.bin").exists()
    )
    return has_config and has_weights


def _iter_qwen_adapter_candidates() -> Iterable[Path]:
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
        for adapter in sorted(
            root.glob("**/adapter"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            yield adapter
        for ckpt in sorted(
            root.glob("**/checkpoint-*"), key=lambda p: p.stat().st_mtime, reverse=True
        ):
            yield ckpt


def _resolve_qwen_adapter_dir(adapter_dir: Optional[str]) -> Path:
    if adapter_dir:
        p = Path(adapter_dir)
        if _is_valid_qwen_adapter_dir(p):
            return p
        raise FileNotFoundError(
            f"Qwen adapter 目录无效: {p}\n"
            "需要包含 adapter_config.json 和 adapter_model 权重文件。"
        )

    for candidate in _iter_qwen_adapter_candidates():
        if _is_valid_qwen_adapter_dir(candidate):
            return candidate

    raise FileNotFoundError(
        "未找到可用的 Qwen adapter。\n"
        "请先训练或下载 adapter，再通过 --adapter_dir 指定。\n"
        "训练示例:\n"
        "  python nlp/qwen/qwen_sft_parser/train.py "
        "--base_model Qwen/Qwen3-1.7B-Base --dataset_dir data/qwen_sft_parser "
        "--output_root outputs/qwen_sft_parser"
    )


class BackendParser:
    """Stateful parser wrapper that can serve rule / NER / Qwen backends."""

    def __init__(
        self,
        backend: str = "rule",
        model_dir: str = "outputs/media_filename_ner",
        max_length: int = 192,
        base_model: str = "Qwen/Qwen3-1.7B-Base",
        adapter_dir: Optional[str] = None,
        device: Optional[str] = None,
        merge_lora: bool = False,
        strict_json: bool = False,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.85,
        repetition_penalty: float = 1.05,
        show_raw_generation: bool = False,
        path_aware_mode: str = "raw_plus_basename",
    ) -> None:
        backend = (backend or "rule").lower().strip()
        if backend not in _BACKEND_CHOICES:
            raise ValueError(
                f"Unsupported backend: {backend}. Choose from {', '.join(_BACKEND_CHOICES)}."
            )

        self.backend = backend
        self.max_length = max_length
        self.device = device

        self.strict_json = strict_json
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.show_raw_generation = show_raw_generation
        self.path_aware_mode = (
            path_aware_mode
            if path_aware_mode in _PATH_AWARE_MODES
            else "raw_plus_basename"
        )

        self._ner_predictor = None
        self._qwen_parser = None

        if self.backend == "ner":
            self._init_ner(model_dir=model_dir)
        elif self.backend == "qwen":
            self._init_qwen(
                base_model=base_model,
                adapter_dir=adapter_dir,
                merge_lora=merge_lora,
            )

    def _init_ner(self, model_dir: str) -> None:
        model_path = Path(model_dir)
        if not _is_valid_ner_model_dir(model_path):
            raise FileNotFoundError(
                f"未找到可用的 NER 模型目录: {model_path}\n"
                "请先训练或下载模型后重试。\n"
                "训练示例:\n"
                "  python nlp/ner/train_media_filename_ner.py "
                "--dataset_path parsed_dataset.json "
                "--output_dir outputs/media_filename_ner "
                "--model_name xlm-roberta-base"
            )

        ner_root = _REPO_ROOT / "nlp" / "ner"
        if str(ner_root) not in sys.path:
            sys.path.insert(0, str(ner_root))

        try:
            from train_media_filename_ner import MediaFilenameNERPredictor  # pylint: disable=import-outside-toplevel
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "NER 后端初始化失败。请确认依赖已安装（torch/transformers）并且代码可导入。"
            ) from exc

        self._ner_predictor = MediaFilenameNERPredictor(
            model_dir_or_name=str(model_path),
            max_length=self.max_length,
            device=self.device,
        )

    def _init_qwen(
        self, base_model: str, adapter_dir: Optional[str], merge_lora: bool
    ) -> None:
        qwen_root = _REPO_ROOT / "nlp" / "qwen"
        if str(qwen_root) not in sys.path:
            sys.path.insert(0, str(qwen_root))

        try:
            from qwen_sft_parser.inference import (  # pylint: disable=import-outside-toplevel
                QwenFilenameJsonParser,
            )
        except Exception as exc:  # pylint: disable=broad-except
            raise RuntimeError(
                "Qwen 后端初始化失败。请确认依赖已安装（torch/transformers/peft）并且代码可导入。"
            ) from exc

        resolved_adapter = _resolve_qwen_adapter_dir(adapter_dir)
        self._qwen_parser = QwenFilenameJsonParser(
            base_model=base_model,
            adapter_dir=str(resolved_adapter),
            device=self.device,
            merge_lora=merge_lora,
            path_aware_mode=self.path_aware_mode,
        )

    def parse(self, path_or_name: str) -> Dict[str, Any]:
        text = (path_or_name or "").strip()
        if not text:
            raise ValueError("Input text is empty.")

        if self.backend == "rule":
            return _clean_rule_result(_parse_filename_rule(text))

        filename = _safe_basename(text)
        if self.backend == "ner":
            ner_input = _compose_path_aware_text(text, mode=self.path_aware_mode)
            parsed = self._ner_predictor.predict(ner_input, return_spans=False)  # type: ignore[union-attr]
            return {
                "raw_path": text,
                "filename": filename,
                "parsed": parsed,
                "confidence": None,
                "backend": "ner",
            }

        qwen_result = self._qwen_parser.parse(  # type: ignore[union-attr]
            path_or_name=text,
            max_new_tokens=self.max_new_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            strict_json=self.strict_json,
        )
        out: Dict[str, Any] = {
            "raw_path": text,
            "filename": filename,
            "parsed": qwen_result.get("parsed", {}),
            "confidence": None,
            "backend": "qwen",
        }
        if "error" in qwen_result:
            out["error"] = qwen_result.get("error")
        if self.show_raw_generation:
            out["raw_generation"] = qwen_result.get("raw_generation", "")
        return out


def create_backend_parser(
    backend: str = "rule",
    model_dir: str = "outputs/media_filename_ner",
    max_length: int = 192,
    base_model: str = "Qwen/Qwen3-1.7B-Base",
    adapter_dir: Optional[str] = None,
    device: Optional[str] = None,
    merge_lora: bool = False,
    strict_json: bool = False,
    max_new_tokens: int = 256,
    temperature: float = 0.1,
    top_p: float = 0.85,
    repetition_penalty: float = 1.05,
    show_raw_generation: bool = False,
    path_aware_mode: str = "raw_plus_basename",
) -> BackendParser:
    return BackendParser(
        backend=backend,
        model_dir=model_dir,
        max_length=max_length,
        base_model=base_model,
        adapter_dir=adapter_dir,
        device=device,
        merge_lora=merge_lora,
        strict_json=strict_json,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        show_raw_generation=show_raw_generation,
        path_aware_mode=path_aware_mode,
    )


def parse_with_backend(
    path_or_name: str, backend: str = "rule", **kwargs: Any
) -> Dict[str, Any]:
    parser = create_backend_parser(backend=backend, **kwargs)
    return parser.parse(path_or_name)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Parse media filenames with selectable backend: rule / ner / qwen."
    )
    parser.add_argument("--backend", type=str, default="rule", choices=_BACKEND_CHOICES)
    parser.add_argument(
        "--text", type=str, default=None, help="Single path/filename to parse."
    )
    parser.add_argument(
        "--input_file", type=str, default="scrape_files_list_merged.txt"
    )
    parser.add_argument("--output_file", type=str, default="parsed_dataset.json")
    parser.add_argument(
        "--low_conf_file",
        type=str,
        default="parsed_dataset_low_confidence.json",
        help="Output path for confidence<1.0 items. For NLP backends this is usually empty.",
    )
    parser.add_argument("--encoding", type=str, default="utf-8")
    parser.add_argument(
        "--pretty", action="store_true", help="Pretty-print single-item JSON output."
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
        choices=_PATH_AWARE_MODES,
        help="Input text composition for NLP backends.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def _load_input_lines(input_file: Path, encoding: str) -> List[str]:
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("r", encoding=encoding) as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def _dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _setup_logging(level_name: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def main() -> None:
    args = _parse_args()
    _setup_logging(args.log_level)
    parser = create_backend_parser(
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

    if args.text is not None:
        result = parser.parse(args.text)
        print(json.dumps(result, ensure_ascii=False, indent=2 if args.pretty else None))
        return

    input_file = Path(args.input_file)
    output_file = Path(args.output_file)
    low_conf_file = Path(args.low_conf_file) if args.low_conf_file else None

    LOGGER.info("Parsing %s with backend=%s ...", input_file, args.backend)
    lines = _load_input_lines(input_file, encoding=args.encoding)

    results: List[Dict[str, Any]] = []
    for i, line in enumerate(lines):
        result = parser.parse(line)
        results.append(result)
        if (i + 1) % 2000 == 0:
            LOGGER.info("Processed %d lines ...", i + 1)

    _dump_json(output_file, results)

    low_conf_results = [
        r
        for r in results
        if isinstance(r.get("confidence"), (int, float))
        and float(r.get("confidence")) < 1.0
    ]
    low_conf_results.sort(key=lambda x: x.get("confidence", 0.0))

    if low_conf_file is not None:
        _dump_json(low_conf_file, low_conf_results)
        LOGGER.info(
            "Found %d items with confidence < 1.0. Saved to %s ...",
            len(low_conf_results),
            low_conf_file,
        )

    if args.backend != "rule":
        LOGGER.warning(
            "NLP backends do not emit media_filename_parser.rules confidence scores by default."
        )
    LOGGER.info("Total processed: %d. Saved to %s ...", len(results), output_file)
    LOGGER.info("Parsing completed.")


if __name__ == "__main__":
    main()
