#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Evaluate trained media filename NER model and generate metric charts.

Usage:
  python nlp/ner/evaluate_model_plots.py
  python nlp/ner/evaluate_model_plots.py --split test --output_dir outputs/media_filename_ner_eval
  python nlp/ner/evaluate_model_plots.py --max_eval_samples 500 --pretty
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from sklearn.metrics import confusion_matrix

from nlp.shared import setup_logging as setup_shared_logging
from nlp.ner.train_media_filename_ner import (
    PARSED_KEY_TO_ENTITY,
    build_dataset_splits,
    load_json_records,
    preprocess_records_to_examples,
    tokenize_and_align_labels,
)


LOGGER = logging.getLogger("media_filename_eval")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate NER model and generate charts."
    )
    parser.add_argument("--dataset_path", type=str, default="parsed_dataset.json")
    parser.add_argument("--model_dir", type=str, default="outputs/media_filename_ner")
    parser.add_argument(
        "--output_dir", type=str, default="outputs/media_filename_ner_eval"
    )
    parser.add_argument(
        "--split", type=str, choices=["train", "validation", "test"], default="test"
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_confidence", type=float, default=0.8)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument(
        "--device", type=str, default=None, help="cuda/cpu, default auto."
    )
    parser.add_argument(
        "--topn_errors", type=int, default=15, help="Top-N misclassified entity pairs."
    )
    parser.add_argument(
        "--include_o_in_topn",
        action="store_true",
        help="Include O in Top-N error-pair stats (default: exclude O).",
    )
    parser.add_argument("--pretty", action="store_true")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def setup_logging(level_name: str) -> None:
    setup_shared_logging(level_name)


def get_device(explicit: str | None, torch_module: Any) -> Any:
    if explicit is not None:
        return torch_module.device(explicit)
    return torch_module.device("cuda" if torch_module.cuda.is_available() else "cpu")


def safe_id_to_label(id2label: Dict[Any, str], idx: int) -> str:
    if idx in id2label:
        return id2label[idx]
    key = str(idx)
    if key in id2label:
        return id2label[key]
    return "O"


def collapse_to_entity(label: str) -> str:
    if label == "O":
        return "O"
    if "-" not in label:
        return label
    return label.split("-", 1)[1]


def to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:  # pylint: disable=broad-except
        return 0.0


def to_int(value: Any) -> int:
    try:
        return int(value)
    except Exception:  # pylint: disable=broad-except
        return 0


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, tuple):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def load_eval_dataset(args: argparse.Namespace) -> Tuple[Any, List[str]]:
    records = load_json_records(Path(args.dataset_path))
    examples, field_stats, global_stats = preprocess_records_to_examples(
        records,
        min_confidence=args.min_confidence,
    )
    LOGGER.info(
        "Preprocess: input=%d kept=%d low_conf=%d",
        global_stats["input_records"],
        global_stats["kept_records"],
        global_stats["skipped_low_conf"],
    )
    LOGGER.info("Field match rate snapshot: %s", field_stats["title"])

    splits = build_dataset_splits(examples, seed=args.seed)
    selected = splits[args.split]

    if args.max_eval_samples is not None and args.max_eval_samples > 0:
        capped = min(args.max_eval_samples, len(selected))
        selected = selected.select(range(capped))
        LOGGER.info("Using first %d samples from split=%s", capped, args.split)

    texts = list(selected["text"])
    return selected, texts


def run_model_predictions(
    dataset_split: Any,
    _texts: Sequence[str],
    args: argparse.Namespace,
) -> Tuple[List[List[str]], List[List[str]], List[str], List[str]]:
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Missing dependency: torch") from exc

    try:
        import evaluate
    except ImportError as exc:
        raise ImportError("Missing dependency: evaluate") from exc

    try:
        from seqeval.metrics import classification_report
    except ImportError as exc:
        raise ImportError("Missing dependency: seqeval") from exc

    try:
        from torch.utils.data import DataLoader
        from transformers import (
            AutoModelForTokenClassification,
            AutoTokenizer,
            DataCollatorForTokenClassification,
        )
    except ImportError as exc:
        raise ImportError("Missing dependency: transformers") from exc

    tokenizer = AutoTokenizer.from_pretrained(args.model_dir, use_fast=True)
    model = AutoModelForTokenClassification.from_pretrained(args.model_dir)
    device = get_device(args.device, torch)
    model.to(device)
    model.eval()
    LOGGER.info("Running on device: %s", device)

    tokenized = dataset_split.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": model.config.label2id,
            "max_length": args.max_length,
        },
        remove_columns=dataset_split.column_names,
        desc="Tokenize + align for eval",
    )

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    loader = DataLoader(
        tokenized, batch_size=args.batch_size, shuffle=False, collate_fn=collator
    )

    all_true_labels: List[List[str]] = []
    all_pred_labels: List[List[str]] = []
    token_true_entities: List[str] = []
    token_pred_entities: List[str] = []

    id2label = model.config.id2label

    for batch in loader:
        labels = batch.pop("labels")
        inputs = {k: v.to(device) for k, v in batch.items()}

        with torch.no_grad():
            logits = model(**inputs).logits
        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        label_ids = labels.cpu().numpy()

        for pred_row, label_row in zip(pred_ids, label_ids):
            seq_true: List[str] = []
            seq_pred: List[str] = []
            for p, t in zip(pred_row, label_row):
                if int(t) == -100:
                    continue
                true_label = safe_id_to_label(id2label, int(t))
                pred_label = safe_id_to_label(id2label, int(p))
                seq_true.append(true_label)
                seq_pred.append(pred_label)
                token_true_entities.append(collapse_to_entity(true_label))
                token_pred_entities.append(collapse_to_entity(pred_label))
            all_true_labels.append(seq_true)
            all_pred_labels.append(seq_pred)

    # Ensure dependencies are used and import is validated.
    _ = evaluate
    _ = classification_report

    return all_true_labels, all_pred_labels, token_true_entities, token_pred_entities


def compute_reports(
    true_labels: Sequence[Sequence[str]],
    pred_labels: Sequence[Sequence[str]],
) -> Tuple[Dict[str, float], Dict[str, Any]]:
    import evaluate
    from seqeval.metrics import classification_report

    seqeval_metric = evaluate.load("seqeval")
    overall_raw = seqeval_metric.compute(
        predictions=list(pred_labels),
        references=list(true_labels),
        zero_division=0,
    )
    overall = {
        "precision": to_float(overall_raw.get("overall_precision")),
        "recall": to_float(overall_raw.get("overall_recall")),
        "f1": to_float(overall_raw.get("overall_f1")),
        "accuracy": to_float(overall_raw.get("overall_accuracy")),
    }

    report = classification_report(
        list(true_labels),
        list(pred_labels),
        digits=6,
        output_dict=True,
        zero_division=0,
    )
    return overall, report


def plot_overall_metrics(overall: Dict[str, float], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    labels = ["precision", "recall", "f1", "accuracy"]
    values = [overall[k] for k in labels]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Overall NER Metrics")
    ax.set_ylabel("Score")
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0, val + 0.01, f"{val:.4f}", ha="center"
        )
    fig.tight_layout()
    fig.savefig(output_dir / "overall_metrics.png", dpi=200)
    plt.close(fig)


def plot_per_entity_metrics(report: Dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    entities = list(PARSED_KEY_TO_ENTITY.values())
    precision = [
        to_float(report.get(entity, {}).get("precision")) for entity in entities
    ]
    recall = [to_float(report.get(entity, {}).get("recall")) for entity in entities]
    f1 = [to_float(report.get(entity, {}).get("f1-score")) for entity in entities]

    x = np.arange(len(entities))
    width = 0.25

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x - width, precision, width=width, label="Precision")
    ax.bar(x, recall, width=width, label="Recall")
    ax.bar(x + width, f1, width=width, label="F1")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(entities, rotation=30, ha="right")
    ax.set_title("Per-Entity Metrics")
    ax.set_ylabel("Score")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "per_entity_metrics.png", dpi=200)
    plt.close(fig)


def plot_entity_support(report: Dict[str, Any], output_dir: Path) -> None:
    import matplotlib.pyplot as plt

    entities = list(PARSED_KEY_TO_ENTITY.values())
    supports = [to_int(report.get(entity, {}).get("support")) for entity in entities]

    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(entities, supports)
    ax.set_title("Entity Support (Test Tokens)")
    ax.set_ylabel("Token Count")
    ax.set_xticks(np.arange(len(entities)))
    ax.set_xticklabels(entities, rotation=30, ha="right")
    for bar, val in zip(bars, supports):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            val,
            str(val),
            ha="center",
            va="bottom",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(output_dir / "entity_support.png", dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    true_entities: Sequence[str],
    pred_entities: Sequence[str],
    output_dir: Path,
) -> None:
    import matplotlib.pyplot as plt

    labels = ["O"] + list(PARSED_KEY_TO_ENTITY.values())
    cm = confusion_matrix(true_entities, pred_entities, labels=labels)

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", aspect="auto")
    ax.figure.colorbar(im, ax=ax)
    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True Entity",
        xlabel="Pred Entity",
        title="Token-Level Entity Confusion Matrix",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    threshold = cm.max() / 2.0 if cm.size else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=7,
            )

    fig.tight_layout()
    fig.savefig(output_dir / "token_entity_confusion_matrix.png", dpi=220)
    plt.close(fig)


def compute_top_error_pairs(
    true_entities: Sequence[str],
    pred_entities: Sequence[str],
    topn: int,
    include_o: bool,
) -> List[Dict[str, Any]]:
    pair_counter: Counter[Tuple[str, str]] = Counter()
    for t, p in zip(true_entities, pred_entities):
        if t == p:
            continue
        if not include_o and (t == "O" or p == "O"):
            continue
        pair_counter[(t, p)] += 1

    top_items = pair_counter.most_common(max(1, topn))
    return [
        {
            "true_entity": true_entity,
            "pred_entity": pred_entity,
            "count": int(count),
            "pair": f"{true_entity} -> {pred_entity}",
        }
        for (true_entity, pred_entity), count in top_items
    ]


def plot_top_error_pairs(
    top_pairs: Sequence[Dict[str, Any]], output_dir: Path, topn: int
) -> None:
    import matplotlib.pyplot as plt

    if not top_pairs:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(
            0.5, 0.5, "No misclassified entity pairs found.", ha="center", va="center"
        )
        ax.set_axis_off()
        fig.tight_layout()
        fig.savefig(output_dir / "top_error_entity_pairs.png", dpi=200)
        plt.close(fig)
        return

    labels = [item["pair"] for item in top_pairs]
    counts = [to_int(item["count"]) for item in top_pairs]

    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(12, max(5, 0.45 * len(labels))))
    bars = ax.barh(y, counts)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.invert_yaxis()
    ax.set_xlabel("Token Count")
    ax.set_title(f"Top-{topn} Most Frequent Misclassified Entity Pairs")
    for bar, val in zip(bars, counts):
        ax.text(
            bar.get_width() + 0.5,
            bar.get_y() + bar.get_height() / 2.0,
            str(val),
            va="center",
        )
    fig.tight_layout()
    fig.savefig(output_dir / "top_error_entity_pairs.png", dpi=220)
    plt.close(fig)


def save_reports(
    overall: Dict[str, float],
    report: Dict[str, Any],
    top_error_pairs: Sequence[Dict[str, Any]],
    args: argparse.Namespace,
    sample_count: int,
    output_dir: Path,
) -> None:
    payload = {
        "model_dir": args.model_dir,
        "dataset_path": args.dataset_path,
        "split": args.split,
        "min_confidence": args.min_confidence,
        "seed": args.seed,
        "sample_count": sample_count,
        "overall": overall,
        "per_entity": report,
        "top_error_entity_pairs": list(top_error_pairs),
    }
    payload = to_jsonable(payload)
    report_path = output_dir / "metrics_report.json"
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2 if args.pretty else None)
    LOGGER.info("Saved metric report to %s", report_path)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_split, texts = load_eval_dataset(args)
    LOGGER.info("Evaluating split=%s size=%d", args.split, len(dataset_split))

    true_labels, pred_labels, true_entities, pred_entities = run_model_predictions(
        dataset_split=dataset_split,
        _texts=texts,
        args=args,
    )
    overall, report = compute_reports(true_labels, pred_labels)
    top_error_pairs = compute_top_error_pairs(
        true_entities=true_entities,
        pred_entities=pred_entities,
        topn=args.topn_errors,
        include_o=args.include_o_in_topn,
    )

    save_reports(
        overall,
        report,
        top_error_pairs,
        args,
        sample_count=len(dataset_split),
        output_dir=output_dir,
    )

    plot_overall_metrics(overall, output_dir)
    plot_per_entity_metrics(report, output_dir)
    plot_entity_support(report, output_dir)
    plot_confusion_matrix(true_entities, pred_entities, output_dir)
    plot_top_error_pairs(top_error_pairs, output_dir, topn=args.topn_errors)

    LOGGER.info("Overall: %s", overall)
    if top_error_pairs:
        LOGGER.info("Top error pair: %s", top_error_pairs[0])
    LOGGER.info("All charts saved under %s", output_dir)


if __name__ == "__main__":
    main()
