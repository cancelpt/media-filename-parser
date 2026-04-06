#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Modules 2+3: model loading, LoRA configuration, and SFTTrainer pipeline.

Features:
1. LoRA SFT fine-tuning on Qwen causal language models
2. Assistant-only loss masking (system/user tokens set to -100)
3. Unique run-directory creation to avoid output path collisions
"""

from __future__ import annotations

import argparse
import inspect
import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
from datasets import DatasetDict, load_dataset
from peft import LoraConfig, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    TrainingArguments,
    set_seed,
)
from trl import SFTTrainer


LOGGER = logging.getLogger("qwen_sft_train")

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="LoRA SFT training for filename Text-to-JSON."
    )
    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--dataset_dir", type=str, default="data/qwen_sft_parser_v2")
    parser.add_argument("--output_root", type=str, default="outputs/qwen_sft_parser")
    parser.add_argument("--run_name", type=str, default="qwen3_1p7b_sft_lora")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--no_gradient_checkpointing", action="store_true")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)

    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    parser.add_argument("--trust_remote_code", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--report_to", type=str, default="none")
    return parser.parse_args()


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def make_unique_run_dir(output_root: Path, run_name: str) -> Path:
    """
    Create a unique run directory under output_root.
    """
    output_root.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = output_root / f"{run_name}_{timestamp}"
    if not base.exists():
        base.mkdir(parents=True, exist_ok=True)
        return base

    idx = 2
    while True:
        candidate = output_root / f"{run_name}_{timestamp}_v{idx}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=True)
            return candidate
        idx += 1


def pick_torch_dtype() -> torch.dtype:
    """
    Select dtype by hardware capability: bf16 > fp16 > fp32.
    """
    if torch.cuda.is_available():
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    return torch.float32


def load_jsonl_dataset(dataset_dir: Path) -> DatasetDict:
    train_path = dataset_dir / "train.jsonl"
    valid_path = dataset_dir / "validation.jsonl"
    if not train_path.exists() or not valid_path.exists():
        raise FileNotFoundError(
            f"Missing train/validation jsonl in {dataset_dir}. Please run data_prep.py first."
        )
    ds = load_dataset(
        "json",
        data_files={"train": str(train_path), "validation": str(valid_path)},
    )
    return ds


def build_tokenize_fn(tokenizer: Any, max_length: int):
    """
    Convert messages into input_ids/labels with assistant-only loss masking.
    """

    def _tokenize(batch: Dict[str, List[Any]]) -> Dict[str, List[List[int]]]:
        all_input_ids: List[List[int]] = []
        all_attention_mask: List[List[int]] = []
        all_labels: List[List[int]] = []

        for messages in batch["messages"]:
            if not isinstance(messages, list) or len(messages) < 3:
                continue

            # Full conversation including assistant response.
            full_text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            # Prompt context uses only system+user messages.
            prompt_text = tokenizer.apply_chat_template(
                messages[:-1],
                tokenize=False,
                add_generation_prompt=True,
            )

            full_enc = tokenizer(
                full_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )
            prompt_enc = tokenizer(
                prompt_text,
                truncation=True,
                max_length=max_length,
                add_special_tokens=False,
            )

            input_ids = list(full_enc["input_ids"])
            attention_mask = list(full_enc["attention_mask"])
            labels = input_ids.copy()

            prompt_len = min(len(prompt_enc["input_ids"]), len(labels))
            for i in range(prompt_len):
                labels[i] = -100

            # Skip samples where assistant tokens are fully truncated.
            if not any(token_id != -100 for token_id in labels):
                continue

            all_input_ids.append(input_ids)
            all_attention_mask.append(attention_mask)
            all_labels.append(labels)

        return {
            "input_ids": all_input_ids,
            "attention_mask": all_attention_mask,
            "labels": all_labels,
        }

    return _tokenize


def build_training_args(
    args: argparse.Namespace, checkpoint_dir: Path
) -> TrainingArguments:
    """
    Handle TrainingArguments compatibility across transformers versions.
    """
    init_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_key = (
        "eval_strategy" if "eval_strategy" in init_params else "evaluation_strategy"
    )

    use_gc = args.gradient_checkpointing and not args.no_gradient_checkpointing
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    kwargs: Dict[str, Any] = {
        "output_dir": str(checkpoint_dir),
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.per_device_train_batch_size,
        "per_device_eval_batch_size": args.per_device_eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": args.weight_decay,
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_loss",
        "greater_is_better": False,
        "save_total_limit": args.save_total_limit,
        "report_to": args.report_to,
        "bf16": use_bf16,
        "fp16": use_fp16,
        "gradient_checkpointing": use_gc,
        "dataloader_num_workers": args.dataloader_num_workers,
        "remove_unused_columns": False,
        "warmup_ratio": args.warmup_ratio,
        "lr_scheduler_type": "cosine",
    }
    kwargs[eval_key] = "epoch"
    return TrainingArguments(**kwargs)


def main() -> None:
    args = parse_args()
    setup_logging()
    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    dataset_dir = Path(args.dataset_dir)
    output_root = Path(args.output_root)
    run_dir = make_unique_run_dir(output_root, args.run_name)
    checkpoint_dir = run_dir / "checkpoints"
    adapter_dir = run_dir / "adapter"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info("Run dir: %s", run_dir)

    ds = load_jsonl_dataset(dataset_dir)
    if args.max_train_samples is not None and args.max_train_samples > 0:
        ds["train"] = ds["train"].select(
            range(min(args.max_train_samples, len(ds["train"])))
        )
    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        ds["validation"] = ds["validation"].select(
            range(min(args.max_eval_samples, len(ds["validation"])))
        )

    LOGGER.info(
        "Dataset sizes: train=%d valid=%d", len(ds["train"]), len(ds["validation"])
    )

    dtype = pick_torch_dtype()
    LOGGER.info("Loading base model: %s | dtype=%s", args.base_model, dtype)

    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=args.trust_remote_code,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=args.trust_remote_code,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.use_cache = False

    use_gc = args.gradient_checkpointing and not args.no_gradient_checkpointing
    if use_gc:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        target_modules=LORA_TARGET_MODULES,
    )

    tokenize_fn = build_tokenize_fn(tokenizer, args.max_length)
    tokenized_ds = ds.map(
        tokenize_fn,
        batched=True,
        remove_columns=ds["train"].column_names,
        desc="Tokenizing and building assistant-only labels",
    )
    LOGGER.info(
        "Tokenized sizes: train=%d valid=%d",
        len(tokenized_ds["train"]),
        len(tokenized_ds["validation"]),
    )

    if len(tokenized_ds["train"]) == 0:
        raise RuntimeError("训练集为空，请检查 max_length 或数据构建逻辑。")
    if len(tokenized_ds["validation"]) == 0:
        raise RuntimeError("验证集为空，请检查 max_length 或数据构建逻辑。")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True,
        label_pad_token_id=-100,
        return_tensors="pt",
        pad_to_multiple_of=8 if torch.cuda.is_available() else None,
    )

    training_args = build_training_args(args, checkpoint_dir=checkpoint_dir)

    trainer_kwargs: Dict[str, Any] = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_ds["train"],
        "eval_dataset": tokenized_ds["validation"],
        "data_collator": data_collator,
        "peft_config": lora_config,
    }
    sft_init_params = set(inspect.signature(SFTTrainer.__init__).parameters.keys())
    if "processing_class" in sft_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    elif "tokenizer" in sft_init_params:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = SFTTrainer(**trainer_kwargs)

    LOGGER.info("Start training...")
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    LOGGER.info("Saving LoRA adapter to %s", adapter_dir)
    trainer.model.save_pretrained(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Persist compact run metadata.
    run_meta = {
        "base_model": args.base_model,
        "dataset_dir": str(dataset_dir),
        "run_dir": str(run_dir),
        "checkpoint_dir": str(checkpoint_dir),
        "adapter_dir": str(adapter_dir),
        "train_size": len(tokenized_ds["train"]),
        "valid_size": len(tokenized_ds["validation"]),
        "lora": {
            "r": args.lora_r,
            "alpha": args.lora_alpha,
            "dropout": args.lora_dropout,
            "target_modules": LORA_TARGET_MODULES,
        },
        "train_args": {
            "learning_rate": args.learning_rate,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "num_train_epochs": args.num_train_epochs,
            "max_length": args.max_length,
            "gradient_checkpointing": use_gc,
        },
    }
    with (run_dir / "run_meta.json").open("w", encoding="utf-8") as f:
        json.dump(run_meta, f, ensure_ascii=False, indent=2)

    LOGGER.info("Training done. adapter_dir=%s", adapter_dir)


if __name__ == "__main__":
    main()
