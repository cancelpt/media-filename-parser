#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Module 4: Qwen LoRA inference and output post-processing.

This script:
1. Loads base model and LoRA adapter weights
2. Builds chat prompts and generates model output
3. Extracts and validates JSON objects from generated text
4. Returns an empty structured object in non-strict mode on parse failure
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


LOGGER = logging.getLogger("qwen_sft_inference")
PATH_AWARE_MODES = ("basename", "raw_path", "raw_plus_basename")

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

OUTPUT_TEMPLATE = {k: None for k in FIELDS}

SYSTEM_PROMPT = (
    "你是一个影视文件名结构化解析引擎。"
    "任务：从用户输入的单个文件名中抽取实体。"
    "你必须只输出一个 JSON 对象，不要输出解释、前后缀文本或 Markdown。"
    "输出必须且仅包含这些键：title, zh_title, year, season_episode, resolution, frame_rate, source, video_codec, video_hdr, audio_codec, group。"
    "缺失或不确定的信息必须输出 null，不要猜测。"
    "season_episode 归一化规则：优先 SxxExx；仅季用 Sxx；仅集用 Exx；无法确定用 null。"
    "Keep output strict JSON only."
)


class JsonExtractError(RuntimeError):
    """Raised when JSON extraction/parsing fails for generated text."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inference for Qwen LoRA filename parser."
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument(
        "--adapter_dir",
        type=str,
        required=True,
        help="LoRA adapter path from train.py output.",
    )
    parser.add_argument(
        "--text", type=str, required=True, help="Raw filename or path string."
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
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.85)
    parser.add_argument("--repetition_penalty", type=float, default=1.05)
    parser.add_argument(
        "--merge_lora",
        action="store_true",
        help="Merge LoRA weights into the base model after loading.",
    )
    parser.add_argument(
        "--strict_json",
        action="store_true",
        help="Raise an exception on JSON extraction failure; otherwise return an empty structured output.",
    )
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def safe_basename(path_or_name: str) -> str:
    text = (path_or_name or "").strip()
    if not text:
        return ""
    parts = re.split(r"[\\/]+", text)
    return parts[-1] if parts else text


def compose_path_aware_text(path_or_name: str, mode: str = "raw_plus_basename") -> str:
    text = (path_or_name or "").strip()
    if not text:
        return ""
    basename = safe_basename(text)
    mode = (mode or "raw_plus_basename").strip().lower()
    if mode not in PATH_AWARE_MODES:
        mode = "raw_plus_basename"
    if mode == "basename":
        return basename
    if mode == "raw_path":
        return text
    return f"{text} ||| {basename}"


def normalize_season_episode(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    m = re.search(r"(?i)s\s*0*(\d{1,3})\s*e\s*0*(\d{1,4})", text)
    if m:
        s, e = int(m.group(1)), int(m.group(2))
        return f"S{s:02d}E{e:02d}"

    m = re.search(r"(?i)\b0*(\d{1,3})\s*x\s*0*(\d{1,4})\b", text)
    if m:
        s, e = int(m.group(1)), int(m.group(2))
        return f"S{s:02d}E{e:02d}"

    m = re.search(r"(?i)\be\s*0*(\d{1,4})\b", text)
    if m:
        e = int(m.group(1))
        return f"S01E{e:02d}"

    if re.fullmatch(r"\d{1,4}", text):
        e = int(text)
        return f"S01E{e:02d}"

    return text


def normalize_frame_rate(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None

    m = re.search(r"(?i)\b(\d{2,3}(?:\.\d{1,3})?)\s*fps\b", text)
    if m:
        return f"{m.group(1)}fps"

    m = re.fullmatch(r"(\d{2,3}(?:\.\d{1,3})?)", text)
    if m:
        return f"{m.group(1)}fps"

    return text


def empty_output() -> Dict[str, Optional[str]]:
    return {k: None for k in FIELDS}


def sanitize_output_dict(data: Dict[str, Any]) -> Dict[str, Optional[str]]:
    out = empty_output()
    for field in FIELDS:
        val = data.get(field)
        out[field] = val if val is not None else None
    out["season_episode"] = normalize_season_episode(out["season_episode"])
    out["frame_rate"] = normalize_frame_rate(out["frame_rate"])
    return out


class QwenFilenameJsonParser:
    def __init__(
        self,
        base_model: str,
        adapter_dir: str,
        device: Optional[str] = None,
        merge_lora: bool = True,
        path_aware_mode: str = "raw_plus_basename",
    ) -> None:
        self.base_model_name = base_model
        self.adapter_dir = adapter_dir
        self.device = torch.device(
            device
            if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.path_aware_mode = (
            path_aware_mode
            if path_aware_mode in PATH_AWARE_MODES
            else "raw_plus_basename"
        )

        dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else (torch.float16 if torch.cuda.is_available() else torch.float32)
        )

        # Prefer adapter-local tokenizer config when adapter_dir is available.
        tokenizer_source = adapter_dir if Path(adapter_dir).exists() else base_model
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source, use_fast=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=dtype,
        )
        model = PeftModel.from_pretrained(base, adapter_dir)
        if merge_lora and hasattr(model, "merge_and_unload"):
            model = model.merge_and_unload()

        model.config.pad_token_id = self.tokenizer.pad_token_id
        model.config.eos_token_id = self.tokenizer.eos_token_id
        model.to(self.device)
        model.eval()
        self.model = model

    def _build_messages(self, path_or_name: str) -> list[dict[str, str]]:
        filename = safe_basename(path_or_name)
        model_input_text = compose_path_aware_text(
            path_or_name, mode=self.path_aware_mode
        )
        schema_example = json.dumps(OUTPUT_TEMPLATE, ensure_ascii=False)
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"INPUT: {model_input_text}\\n"
                    f"BASENAME: {filename}\\n"
                    f"Return strict JSON only. Template: {schema_example}"
                ),
            },
        ]

    def _extract_json(self, text: str) -> Dict[str, Any]:
        # Use non-greedy matching and attempt JSON parse for each candidate block.
        candidates = re.findall(r"\{.*?\}", text, flags=re.DOTALL)
        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    return parsed
            except json.JSONDecodeError:
                continue
        raise JsonExtractError("无法从生成文本中提取有效 JSON。")

    def parse(
        self,
        path_or_name: str,
        max_new_tokens: int = 256,
        temperature: float = 0.1,
        top_p: float = 0.85,
        repetition_penalty: float = 1.05,
        strict_json: bool = False,
    ) -> Dict[str, Any]:
        messages = self._build_messages(path_or_name)
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        model_inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only newly generated tokens, excluding the prompt tokens.
        prompt_len = model_inputs["input_ids"].shape[1]
        gen_ids = generated[0][prompt_len:]
        gen_text = self.tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

        try:
            parsed = self._extract_json(gen_text)
            safe = sanitize_output_dict(parsed)
            return {"parsed": safe, "raw_generation": gen_text}
        except JsonExtractError:
            if strict_json:
                raise
            return {
                "parsed": empty_output(),
                "raw_generation": gen_text,
                "error": "json_parse_failed",
            }


def main() -> None:
    args = parse_args()
    setup_logging()

    parser = QwenFilenameJsonParser(
        base_model=args.base_model,
        adapter_dir=args.adapter_dir,
        device=args.device,
        merge_lora=args.merge_lora,
        path_aware_mode=args.path_aware_mode,
    )
    result = parser.parse(
        path_or_name=args.text,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        strict_json=args.strict_json,
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
