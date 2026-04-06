#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train and serve a BIO-based NER model for parsing media filenames.

This script is organized into five modules requested by the user:
1) Data preprocessing + character-level BIO annotation
2) Tokenization + label alignment
3) Dataset construction + model initialization
4) Training + evaluation
5) Inference pipeline (BIO -> structured dict)

Example:
    python nlp/ner/train_media_filename_ner.py ^
        --dataset_path parsed_dataset.json ^
        --output_dir outputs/media_ner ^
        --model_name xlm-roberta-base ^
        --max_train_samples 4000
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch

from nlp.shared import (
    PATH_AWARE_MODES,
    compose_path_aware_text,
    setup_logging as setup_shared_logging,
)


LOGGER = logging.getLogger("media_filename_ner")
MODEL_TYPE_CHOICES = ("plain", "crf")


# ============================================================================
# Global Label Definitions
# ============================================================================

PARSED_KEY_TO_ENTITY: Dict[str, str] = {
    "title": "TITLE",
    "zh_title": "ZH_TITLE",
    "year": "YEAR",
    "season_episode": "SEASON_EPISODE",
    "resolution": "RESOLUTION",
    "frame_rate": "FRAME_RATE",
    "source": "SOURCE",
    "video_codec": "VIDEO_CODEC",
    "video_hdr": "VIDEO_HDR",
    "audio_codec": "AUDIO_CODEC",
    "group": "GROUP",
}
ENTITY_TO_PARSED_KEY: Dict[str, str] = {v: k for k, v in PARSED_KEY_TO_ENTITY.items()}

# Priority is important:
# - metadata fields are usually shorter/more specific and should be placed first
# - title fields are broader and likely to overlap with other values
ENTITY_FIELD_PRIORITY: List[str] = [
    "group",
    "resolution",
    "frame_rate",
    "source",
    "video_codec",
    "video_hdr",
    "audio_codec",
    "year",
    "season_episode",
    "zh_title",
    "title",
]

# Fields that are more likely to appear near the filename tail.
RIGHT_BIASED_FIELDS = {
    "group",
    "resolution",
    "frame_rate",
    "source",
    "video_codec",
    "video_hdr",
    "audio_codec",
}


def build_bio_label_list() -> List[str]:
    labels = ["O"]
    for entity in PARSED_KEY_TO_ENTITY.values():
        labels.append(f"B-{entity}")
        labels.append(f"I-{entity}")
    return labels


# ============================================================================
# Module 1: Data preprocessing and BIO generation
# ============================================================================


@dataclass(frozen=True)
class MatchedSpan:
    """A matched entity span in raw filename character offsets [start, end)."""

    start: int
    end: int
    field: str
    entity: str
    value: str
    strategy: str


def normalize_with_mapping(text: str, compact: bool = False) -> Tuple[str, List[int]]:
    """
    Normalize text with NFKC + lowercase, optionally removing non-alnum chars.
    Returns:
        normalized_text, mapping (normalized_index -> original_char_index)
    """
    out_chars: List[str] = []
    mapping: List[int] = []
    for idx, ch in enumerate(text):
        norm_piece = unicodedata.normalize("NFKC", ch).lower()
        for nch in norm_piece:
            if compact and not nch.isalnum():
                continue
            out_chars.append(nch)
            mapping.append(idx)
    return "".join(out_chars), mapping


def normalize_text(text: str, compact: bool = False) -> str:
    return normalize_with_mapping(text, compact=compact)[0]


def find_all_occurrences(haystack: str, needle: str) -> Iterable[int]:
    """Yield all match positions of needle in haystack, including overlaps."""
    if not needle:
        return
    start = 0
    while True:
        idx = haystack.find(needle, start)
        if idx < 0:
            break
        yield idx
        start = idx + 1


def normalized_span_to_original(
    normalized_start: int,
    normalized_end: int,
    mapping: Sequence[int],
) -> Optional[Tuple[int, int]]:
    if normalized_start < 0 or normalized_end <= normalized_start:
        return None
    if normalized_start >= len(mapping) or normalized_end - 1 >= len(mapping):
        return None
    original_start = mapping[normalized_start]
    original_end = mapping[normalized_end - 1] + 1
    if original_end <= original_start:
        return None
    return original_start, original_end


def dedupe_spans(spans: Iterable[Tuple[int, int]]) -> List[Tuple[int, int]]:
    seen = set()
    out: List[Tuple[int, int]] = []
    for span in spans:
        if span in seen:
            continue
        seen.add(span)
        out.append(span)
    return out


def find_exact_spans(filename: str, value: str) -> List[Tuple[int, int]]:
    """Exact substring search in normalized (NFKC + lower) space."""
    if not value.strip():
        return []
    filename_norm, mapping = normalize_with_mapping(filename, compact=False)
    value_norm = normalize_text(value, compact=False)
    if not value_norm:
        return []

    spans: List[Tuple[int, int]] = []
    for idx in find_all_occurrences(filename_norm, value_norm):
        mapped = normalized_span_to_original(idx, idx + len(value_norm), mapping)
        if mapped:
            spans.append(mapped)
    return dedupe_spans(spans)


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+|[\u4e00-\u9fff]+")
TOKEN_GAP = r"[\s\.\-_()\[\]{}+/\\|,:;~!@#$%^&*]+"


def find_token_gap_spans(filename: str, value: str) -> List[Tuple[int, int]]:
    """
    Regex-based fuzzy search:
    - split entity value to semantic tokens
    - allow punctuation/space separators between tokens
    """
    value_norm = normalize_text(value, compact=False)
    tokens = TOKEN_PATTERN.findall(value_norm)
    if len(tokens) < 2:
        return []

    pattern = TOKEN_GAP.join(re.escape(tok) for tok in tokens)
    filename_norm, mapping = normalize_with_mapping(filename, compact=False)

    spans: List[Tuple[int, int]] = []
    for m in re.finditer(pattern, filename_norm, flags=re.IGNORECASE):
        mapped = normalized_span_to_original(m.start(), m.end(), mapping)
        if mapped:
            spans.append(mapped)
    return dedupe_spans(spans)


def find_compact_spans(filename: str, value: str) -> List[Tuple[int, int]]:
    """
    Compact fuzzy search:
    - remove all non-alnum chars in both sides
    - search compact value in compact filename
    """
    if not value.strip():
        return []

    filename_compact, mapping = normalize_with_mapping(filename, compact=True)
    value_compact = normalize_text(value, compact=True)
    if not value_compact:
        return []

    spans: List[Tuple[int, int]] = []
    for idx in find_all_occurrences(filename_compact, value_compact):
        mapped = normalized_span_to_original(idx, idx + len(value_compact), mapping)
        if mapped:
            spans.append(mapped)
    return dedupe_spans(spans)


def int_to_chinese(num: int) -> str:
    """
    Integer (1..99) to simple Chinese numerals, used for season/episode fallback.
    """
    if num <= 0:
        return str(num)
    digits = "零一二三四五六七八九"
    if num < 10:
        return digits[num]
    if num < 20:
        return "十" + (digits[num % 10] if num % 10 else "")
    if num < 100:
        tens, ones = divmod(num, 10)
        return digits[tens] + "十" + (digits[ones] if ones else "")
    return str(num)


def parse_season_episode(value: str) -> Optional[Tuple[int, int]]:
    m = re.search(r"[sS](\d{1,3})[eE](\d{1,4})", value)
    if not m:
        return None
    return int(m.group(1)), int(m.group(2))


def find_season_episode_spans(filename: str, value: str) -> List[Tuple[int, int]]:
    """
    Field-specific fallback for season_episode:
    - supports S03E13, 3x13
    - supports Chinese forms like:
      第三季 - 13
      第21季第1集
    """
    parsed = parse_season_episode(value)
    if not parsed:
        return []
    season, episode = parsed
    season_cn = int_to_chinese(season)
    episode_cn = int_to_chinese(episode)

    patterns = [
        rf"s\s*0*{season}\s*e\s*0*{episode}",
        rf"{season}\s*x\s*0*{episode}",
        rf"第\s*0*{season}\s*季\s*[-_. ]*第?\s*0*{episode}\s*[集话話]?",
        rf"第\s*{season_cn}\s*季\s*[-_. ]*第?\s*0*{episode}\s*[集话話]?",
        rf"第\s*0*{season}\s*季\s*[-_. ]*第?\s*{episode_cn}\s*[集话話]?",
        rf"第\s*{season_cn}\s*季\s*[-_. ]*第?\s*{episode_cn}\s*[集话話]?",
    ]

    filename_norm, mapping = normalize_with_mapping(filename, compact=False)
    spans: List[Tuple[int, int]] = []
    for p in patterns:
        for m in re.finditer(p, filename_norm, flags=re.IGNORECASE):
            mapped = normalized_span_to_original(m.start(), m.end(), mapping)
            if mapped:
                spans.append(mapped)
    return dedupe_spans(spans)


def unique_preserve_order(values: Iterable[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for val in values:
        cleaned = val.strip()
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        out.append(cleaned)
    return out


def is_hard_negative_candidate(
    text: str,
    parsed: Dict[str, object],
    hard_negative_tokens: Sequence[str],
) -> bool:
    """
    Hard-negative mining for episode confusion cases.
    Focus on rows where season_episode exists and text contains confusing numeric cues.
    """
    season_episode = parsed.get("season_episode")
    if season_episode is None or not str(season_episode).strip():
        return False

    lowered = text.lower()
    token_hit = any(
        tok and str(tok).strip().lower() in lowered for tok in hard_negative_tokens
    )
    has_bracket_number = re.search(r"\[(\d{3,4})\]", text) is not None
    has_year_like = re.search(r"(?<!\d)(19\d{2}|20[0-3]\d)(?!\d)", text) is not None
    has_res_like = (
        re.search(
            r"(?i)(?<!\d)(360|480|540|576|720|900|1080|1440|2160|4320)(?!\d)",
            text,
        )
        is not None
    )
    has_episode_like = (
        re.search(
            r"(?i)\bS\d{1,3}E\d{1,4}\b|\b\d{1,3}x\d{1,4}\b|\[\d{2,4}\]|第.+[季集话話]",
            text,
        )
        is not None
    )
    return bool(
        has_episode_like
        and (token_hit or (has_bracket_number and has_year_like and has_res_like))
    )


def generate_field_candidates(field: str, raw_value: str) -> List[str]:
    """
    Generate candidate surface forms for fuzzy alignment.
    Conservative by design: if uncertain, skip annotation instead of forcing.
    """
    base = str(raw_value).strip()
    candidates: List[str] = [base, re.sub(r"\s+", " ", base)]

    if field in {"title", "zh_title"}:
        # Some parsed values contain a noisy leading index like "2 朋友游戏".
        without_leading_idx = re.sub(r"^\d+\s*[-._:：]*\s*", "", base).strip()
        candidates.append(without_leading_idx)

    if field in {"source", "video_codec", "audio_codec", "frame_rate"}:
        candidates.append(base.replace(" ", "."))
        candidates.append(base.replace(" ", ""))
        candidates.append(base.replace("-", ""))

    if field == "video_codec":
        candidates.append(re.sub(r"(?i)h\.(26[45])", r"H\1", base))
        candidates.append(re.sub(r"(?i)h(26[45])", r"H.\1", base))

    if field == "audio_codec":
        candidates.append(base.replace("(", " ").replace(")", " "))
        candidates.append(base.replace("(", "").replace(")", ""))

    if field == "frame_rate":
        fps_match = re.search(r"(?i)\b(\d{2,3}(?:\.\d{1,3})?)\s*fps\b", base)
        if fps_match:
            fps_num = fps_match.group(1)
            candidates.append(f"{fps_num}fps")
            candidates.append(f"{fps_num} fps")
        elif re.fullmatch(r"\d{2,3}(?:\.\d{1,3})?", base):
            candidates.append(f"{base}fps")
            candidates.append(f"{base} fps")

    return unique_preserve_order(candidates)


def span_is_free(start: int, end: int, occupied: Sequence[bool]) -> bool:
    if start < 0 or end <= start or end > len(occupied):
        return False
    return not any(occupied[start:end])


def select_non_overlapping_span(
    spans: Iterable[Tuple[int, int]],
    occupied: Sequence[bool],
    prefer_right: bool,
) -> Optional[Tuple[int, int]]:
    candidates = dedupe_spans(spans)
    if not candidates:
        return None

    # Prefer longer matches when start ties.
    if prefer_right:
        candidates.sort(key=lambda x: (-x[0], -(x[1] - x[0])))
    else:
        candidates.sort(key=lambda x: (x[0], -(x[1] - x[0])))

    for start, end in candidates:
        if span_is_free(start, end, occupied):
            return start, end
    return None


def match_entity_span(
    filename: str,
    field: str,
    value: str,
    occupied: Sequence[bool],
) -> Optional[Tuple[int, int, str]]:
    """
    Match one field value to a filename span.
    Matching strategy priority:
      1) season-specific regex (for season_episode)
      2) exact normalized search
      3) token-gap fuzzy regex
      4) compact fuzzy search
    """
    candidates = generate_field_candidates(field, value)
    if not candidates:
        return None

    prefer_right = field in RIGHT_BIASED_FIELDS
    base_value = candidates[0]

    if field == "season_episode":
        se_spans = find_season_episode_spans(filename, base_value)
        chosen = select_non_overlapping_span(se_spans, occupied, prefer_right=False)
        if chosen:
            return chosen[0], chosen[1], "season_regex"

    for strategy_name, finder in [
        ("exact", find_exact_spans),
        ("token_gap", find_token_gap_spans),
        ("compact", find_compact_spans),
    ]:
        spans: List[Tuple[int, int]] = []
        for cand in candidates:
            try:
                spans.extend(finder(filename, cand))
            except re.error as err:
                LOGGER.warning("Regex error on field=%s value=%r: %s", field, cand, err)
            except Exception as err:  # pylint: disable=broad-except
                LOGGER.warning("Match error on field=%s value=%r: %s", field, cand, err)
        chosen = select_non_overlapping_span(spans, occupied, prefer_right=prefer_right)
        if chosen:
            return chosen[0], chosen[1], strategy_name

    return None


def apply_span_to_bio(
    labels: List[str], occupied: List[bool], span: MatchedSpan
) -> None:
    labels[span.start] = f"B-{span.entity}"
    for i in range(span.start + 1, span.end):
        labels[i] = f"I-{span.entity}"
    for i in range(span.start, span.end):
        occupied[i] = True


def build_char_bio_labels(
    filename: str, parsed: Dict[str, object]
) -> Tuple[List[str], List[MatchedSpan]]:
    """
    Convert one filename + parsed dict into character-level BIO labels.
    Unmatched entities are skipped and left as O by design.
    """
    labels = ["O"] * len(filename)
    occupied = [False] * len(filename)
    spans: List[MatchedSpan] = []

    for field in ENTITY_FIELD_PRIORITY:
        if field not in PARSED_KEY_TO_ENTITY:
            continue
        value = parsed.get(field)
        if value is None:
            continue
        value_str = str(value).strip()
        if not value_str:
            continue

        match = match_entity_span(filename, field, value_str, occupied)
        if not match:
            continue

        start, end, strategy = match
        entity = PARSED_KEY_TO_ENTITY[field]
        span = MatchedSpan(
            start=start,
            end=end,
            field=field,
            entity=entity,
            value=value_str,
            strategy=strategy,
        )
        apply_span_to_bio(labels, occupied, span)
        spans.append(span)

    return labels, spans


def load_json_records(dataset_path: Path) -> List[Dict[str, object]]:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset file not found: {dataset_path}")
    with dataset_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Dataset root must be a JSON list.")
    return data


def preprocess_records_to_examples(
    records: Sequence[Dict[str, object]],
    min_confidence: float = 0.8,
    path_aware_mode: str = "raw_plus_basename",
    hard_negative_boost: int = 1,
    hard_negative_tokens: Sequence[str] = ("2024", "1080", "1134"),
) -> Tuple[List[Dict[str, object]], Dict[str, Dict[str, int]], Dict[str, int]]:
    """
    Build trainable examples:
        {"text": filename, "char_labels": [...BIO labels...]}
    Also returns per-field matching stats for observability.
    """
    examples: List[Dict[str, object]] = []
    field_stats: Dict[str, Dict[str, int]] = {
        field: {"total_non_null": 0, "matched": 0} for field in PARSED_KEY_TO_ENTITY
    }
    global_stats = {
        "input_records": len(records),
        "kept_records": 0,
        "skipped_low_conf": 0,
        "skipped_bad_record": 0,
        "hard_negative_candidates": 0,
        "hard_negative_added": 0,
    }
    hard_negative_examples: List[Dict[str, object]] = []

    for idx, item in enumerate(records):
        try:
            confidence = float(item.get("confidence", 0.0) or 0.0)
        except Exception:  # pylint: disable=broad-except
            confidence = 0.0

        if confidence < min_confidence:
            global_stats["skipped_low_conf"] += 1
            continue

        raw_path = item.get("raw_path") if isinstance(item.get("raw_path"), str) else ""
        filename = item.get("filename")
        parsed = item.get("parsed")
        if not isinstance(filename, str):
            filename = ""
        model_text = compose_path_aware_text(
            raw_path=raw_path,
            filename=filename,
            mode=path_aware_mode,
        )
        if not model_text:
            global_stats["skipped_bad_record"] += 1
            continue
        if not isinstance(parsed, dict):
            parsed = {}

        # Count non-null entities first (for match-rate reporting).
        for field in PARSED_KEY_TO_ENTITY:
            value = parsed.get(field)
            if value is None:
                continue
            if str(value).strip():
                field_stats[field]["total_non_null"] += 1

        labels, spans = build_char_bio_labels(model_text, parsed)
        if len(labels) != len(model_text):
            # Defensive fallback: if anything goes wrong, keep all-O labels.
            LOGGER.warning("Label length mismatch at index=%d; fallback to all O.", idx)
            labels = ["O"] * len(model_text)
            spans = []

        for span in spans:
            field_stats[span.field]["matched"] += 1

        example = {"text": model_text, "char_labels": labels}
        examples.append(example)
        if is_hard_negative_candidate(model_text, parsed, hard_negative_tokens):
            hard_negative_examples.append({"text": model_text, "char_labels": labels})
        global_stats["kept_records"] += 1

    global_stats["hard_negative_candidates"] = len(hard_negative_examples)
    if hard_negative_boost > 1 and hard_negative_examples:
        boosted: List[Dict[str, object]] = []
        for ex in hard_negative_examples:
            for _ in range(hard_negative_boost - 1):
                boosted.append(
                    {"text": ex["text"], "char_labels": list(ex["char_labels"])}
                )
        examples.extend(boosted)
        global_stats["hard_negative_added"] = len(boosted)

    return examples, field_stats, global_stats


# ============================================================================
# Module 2: Tokenization and label alignment
# ============================================================================


def align_char_labels_to_token_labels(
    texts: Sequence[str],
    char_labels_batch: Sequence[Sequence[str]],
    tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int,
) -> Dict[str, List[List[int]]]:
    """
    Tokenize raw text and align char-level BIO labels to token-level labels.
    Rules:
      - Special tokens ([CLS], [SEP], etc.) => -100
      - Token label is inferred from covered characters
      - If token starts inside an entity with I-*, but it is the first token of that
        contiguous span, we normalize it to B-* to keep valid BIO.
    """
    tokenized = tokenizer(
        list(texts),
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
    )

    all_labels: List[List[int]] = []
    for i, offsets in enumerate(tokenized["offset_mapping"]):
        text = texts[i]
        char_labels = char_labels_batch[i]
        if len(char_labels) != len(text):
            # Safety fallback for malformed sample.
            char_labels = ["O"] * len(text)

        label_ids: List[int] = []
        prev_entity: Optional[str] = None
        prev_end = -1

        for start, end in offsets:
            if start == end:
                # Special token.
                label_ids.append(-100)
                prev_entity = None
                prev_end = -1
                continue

            if start >= len(char_labels):
                label_ids.append(label2id["O"])
                prev_entity = None
                prev_end = end
                continue

            end = min(end, len(char_labels))
            token_span_labels = [lab for lab in char_labels[start:end] if lab != "O"]
            if not token_span_labels:
                label_ids.append(label2id["O"])
                prev_entity = None
                prev_end = end
                continue

            first = token_span_labels[0]
            if "-" not in first:
                label_ids.append(label2id["O"])
                prev_entity = None
                prev_end = end
                continue

            prefix, entity = first.split("-", 1)
            if prefix == "B":
                normalized_prefix = "B"
            else:
                # Convert orphan I-* to B-*, keep I-* only for contiguous continuation.
                is_continuation = prev_entity == entity and start <= prev_end
                normalized_prefix = "I" if is_continuation else "B"

            normalized_label = f"{normalized_prefix}-{entity}"
            label_ids.append(label2id.get(normalized_label, label2id["O"]))
            prev_entity = entity
            prev_end = end

        all_labels.append(label_ids)

    tokenized["labels"] = all_labels
    tokenized.pop("offset_mapping")
    return tokenized


def tokenize_and_align_labels(
    examples: Dict[str, List[object]],
    tokenizer: Any,
    label2id: Dict[str, int],
    max_length: int,
) -> Dict[str, List[List[int]]]:
    return align_char_labels_to_token_labels(
        texts=examples["text"],
        char_labels_batch=examples["char_labels"],
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
    )


# ============================================================================
# Module 3: Dataset construction and model initialization
# ============================================================================


def build_dataset_splits(examples: Sequence[Dict[str, object]], seed: int = 42) -> Any:
    if len(examples) < 20:
        raise ValueError(
            f"Need at least 20 examples to create stable train/val/test splits, got {len(examples)}."
        )
    try:
        from datasets import Dataset, DatasetDict
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: datasets. Please install with `pip install datasets`."
        ) from exc

    ds = Dataset.from_list(list(examples))
    train_vs_rest = ds.train_test_split(test_size=0.2, seed=seed, shuffle=True)
    val_vs_test = train_vs_rest["test"].train_test_split(
        test_size=0.5, seed=seed, shuffle=True
    )
    return DatasetDict(
        {
            "train": train_vs_rest["train"],
            "validation": val_vs_test["train"],
            "test": val_vs_test["test"],
        }
    )


def build_label_mappings() -> Tuple[List[str], Dict[str, int], Dict[int, str]]:
    labels = build_bio_label_list()
    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


class BertCrfForTokenClassification(torch.nn.Module):
    """
    Lightweight BERT+CRF token classification wrapper.

    The class mimics `save_pretrained`/`from_pretrained` usage so it can plug into
    current Trainer + Predictor code paths.
    """

    def __init__(self, encoder: Any, config: Any) -> None:
        super().__init__()
        self.encoder = encoder
        self.config = config
        self.num_labels = int(config.num_labels)

        hidden_dropout = float(getattr(config, "hidden_dropout_prob", 0.1))
        hidden_size = int(getattr(config, "hidden_size"))

        self.dropout = torch.nn.Dropout(hidden_dropout)
        self.classifier = torch.nn.Linear(hidden_size, self.num_labels)
        try:
            from torchcrf import CRF
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: torchcrf. Please install with `pip install pytorch-crf`."
            ) from exc
        self.crf = CRF(self.num_labels, batch_first=True)

        self.config.use_crf = True
        self.config.model_type_for_ner = "crf"

    @classmethod
    def from_base_model(
        cls,
        model_name: str,
        label2id: Dict[str, int],
        id2label: Dict[int, str],
    ) -> "BertCrfForTokenClassification":
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: transformers. Please install with `pip install transformers`."
            ) from exc

        config = AutoConfig.from_pretrained(model_name)
        config.num_labels = len(label2id)
        config.label2id = label2id
        config.id2label = {int(k): v for k, v in id2label.items()}
        config.use_crf = True
        config.model_type_for_ner = "crf"
        config.base_model_name = model_name
        encoder = AutoModel.from_pretrained(model_name, config=config)
        return cls(encoder=encoder, config=config)

    @classmethod
    def from_pretrained(cls, model_dir_or_name: str) -> "BertCrfForTokenClassification":
        try:
            from transformers import AutoConfig, AutoModel
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: transformers. Please install with `pip install transformers`."
            ) from exc

        path = Path(model_dir_or_name)
        if not path.exists():
            raise FileNotFoundError(f"CRF model directory not found: {path}")

        config = AutoConfig.from_pretrained(str(path))
        encoder = AutoModel.from_config(config)
        model = cls(encoder=encoder, config=config)

        state_path = path / "pytorch_model.bin"
        if state_path.exists():
            state = torch.load(str(state_path), map_location="cpu")
        else:
            safe_path = path / "model.safetensors"
            if not safe_path.exists():
                raise FileNotFoundError(
                    f"Missing CRF state dict: expected {state_path} or {safe_path}"
                )
            try:
                from safetensors.torch import load_file
            except ImportError as exc:
                raise ImportError(
                    "Missing dependency: safetensors. Please install with `pip install safetensors`."
                ) from exc
            state = load_file(str(safe_path))
        model.load_state_dict(state, strict=True)
        return model

    def save_pretrained(self, save_directory: str) -> None:
        save_dir = Path(save_directory)
        save_dir.mkdir(parents=True, exist_ok=True)
        self.encoder.config.use_crf = True
        self.encoder.config.model_type_for_ner = "crf"
        self.encoder.config.save_pretrained(str(save_dir))
        torch.save(self.state_dict(), str(save_dir / "pytorch_model.bin"))

    @staticmethod
    def _decoded_to_one_hot(
        decoded: Sequence[Sequence[int]],
        batch_size: int,
        seq_len: int,
        num_labels: int,
        device: Any,
        dtype: Any,
    ) -> Any:
        import torch

        logits = torch.zeros(
            (batch_size, seq_len, num_labels), device=device, dtype=dtype
        )
        for i, tags in enumerate(decoded):
            if not tags:
                continue
            t = torch.tensor(tags[:seq_len], device=device, dtype=torch.long)
            pos = torch.arange(t.shape[0], device=device)
            logits[i, pos, t] = 1.0
        return logits

    def forward(
        self,
        input_ids: Any = None,
        attention_mask: Any = None,
        labels: Any = None,
        token_type_ids: Any = None,
        **kwargs: Any,
    ) -> Any:
        try:
            from transformers.modeling_outputs import TokenClassifierOutput
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: transformers. Please install with `pip install transformers`."
            ) from exc

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            **kwargs,
        )
        hidden = outputs[0]
        emissions = self.classifier(self.dropout(hidden))

        mask = (
            attention_mask.bool()
            if attention_mask is not None
            else torch.ones_like(input_ids).bool()
        )
        decoded = self.crf.decode(emissions, mask=mask)
        logits = self._decoded_to_one_hot(
            decoded=decoded,
            batch_size=emissions.shape[0],
            seq_len=emissions.shape[1],
            num_labels=emissions.shape[2],
            device=emissions.device,
            dtype=emissions.dtype,
        )

        loss = None
        if labels is not None:
            tags = labels.clone()
            valid_mask = mask & (labels >= 0)
            tags[tags < 0] = 0
            # torchcrf requires the first timestep mask to be on for every sequence.
            # We force step-0 as a valid "O" tag anchor when needed.
            if valid_mask.ndim == 2:
                if valid_mask.shape[1] > 0:
                    valid_mask[:, 0] = True
                    tags[:, 0] = 0
                for i in range(valid_mask.shape[0]):
                    if not bool(valid_mask[i].any()):
                        valid_mask[i, 0] = True
                        tags[i, 0] = 0
            loss = -self.crf(emissions, tags, mask=valid_mask, reduction="mean")

        return TokenClassifierOutput(loss=loss, logits=logits)


def initialize_tokenizer_and_model(
    model_name: str,
    label2id: Dict[str, int],
    id2label: Dict[int, str],
    model_type: str = "plain",
) -> Tuple[Any, Any]:
    try:
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: transformers. Please install with `pip install transformers`."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    model_type = (model_type or "plain").strip().lower()
    if model_type not in MODEL_TYPE_CHOICES:
        raise ValueError(
            f"Unsupported model_type={model_type}. Choices: {MODEL_TYPE_CHOICES}"
        )

    if model_type == "crf":
        model = BertCrfForTokenClassification.from_base_model(
            model_name=model_name,
            label2id=label2id,
            id2label=id2label,
        )
    else:
        model = AutoModelForTokenClassification.from_pretrained(
            model_name,
            num_labels=len(label2id),
            id2label=id2label,
            label2id=label2id,
        )
        model.config.use_crf = False
        model.config.model_type_for_ner = "plain"
    return tokenizer, model


# ============================================================================
# Module 4: Training and evaluation
# ============================================================================


def build_compute_metrics_fn(id2label: Dict[int, str]):
    seqeval_metric = None
    try:
        import evaluate
    except ImportError:
        evaluate = None
    else:
        try:
            seqeval_metric = evaluate.load("seqeval")
        except Exception as exc:  # pylint: disable=broad-except
            LOGGER.warning(
                "evaluate.load('seqeval') is unavailable; falling back to seqeval.metrics. reason=%s",
                exc,
            )
    try:
        import numpy as np
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: numpy. Please install with `pip install numpy`."
        ) from exc
    try:
        from seqeval.metrics import (
            accuracy_score,
            f1_score,
            precision_score,
            recall_score,
        )
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: seqeval. Please install with `pip install seqeval`."
        ) from exc

    def compute_metrics(eval_pred: Tuple[Any, Any]) -> Dict[str, float]:
        logits, labels = eval_pred
        pred_ids = np.argmax(logits, axis=-1)

        true_predictions: List[List[str]] = []
        true_labels: List[List[str]] = []

        for pred_row, label_row in zip(pred_ids, labels):
            current_preds: List[str] = []
            current_labels: List[str] = []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                current_preds.append(id2label[int(pred_id)])
                current_labels.append(id2label[int(label_id)])
            true_predictions.append(current_preds)
            true_labels.append(current_labels)

        if seqeval_metric is not None:
            results = seqeval_metric.compute(
                predictions=true_predictions,
                references=true_labels,
                zero_division=0,
            )
            return {
                "precision": float(results.get("overall_precision", 0.0)),
                "recall": float(results.get("overall_recall", 0.0)),
                "f1": float(results.get("overall_f1", 0.0)),
                "accuracy": float(results.get("overall_accuracy", 0.0)),
            }

        return {
            "precision": float(
                precision_score(true_labels, true_predictions, zero_division=0)
            ),
            "recall": float(
                recall_score(true_labels, true_predictions, zero_division=0)
            ),
            "f1": float(f1_score(true_labels, true_predictions, zero_division=0)),
            "accuracy": float(accuracy_score(true_labels, true_predictions)),
        }

    return compute_metrics


def build_training_arguments(args: argparse.Namespace) -> Any:
    try:
        import torch
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: torch. Please install with `pip install torch`."
        ) from exc
    try:
        import inspect
    except ImportError as exc:
        raise ImportError("Missing dependency: inspect (stdlib).") from exc
    try:
        from transformers import TrainingArguments
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: transformers. Please install with `pip install transformers`."
        ) from exc

    init_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    strategy_key = (
        "eval_strategy" if "eval_strategy" in init_params else "evaluation_strategy"
    )

    kwargs = {
        "output_dir": args.output_dir,
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": args.batch_size,
        "per_device_eval_batch_size": args.batch_size,
        "num_train_epochs": args.num_train_epochs,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "save_total_limit": 2,
        "logging_strategy": "steps",
        "logging_steps": args.logging_steps,
        "report_to": "none",
        "fp16": torch.cuda.is_available(),
        "dataloader_num_workers": args.dataloader_num_workers,
    }
    kwargs[strategy_key] = "epoch"
    return TrainingArguments(**kwargs)


# ============================================================================
# Module 5: Inference pipeline
# ============================================================================


SEPARATOR_ONLY_PATTERN = re.compile(r"^[\s\.\-_()\[\]{}+/\\|,:;~!@#$%^&*]*$")


def canonicalize_snippet(text: str) -> str:
    return text.strip(" \t\r\n.-_()[]{}")


def smart_join_segments(parts: Sequence[str]) -> str:
    clean_parts = []
    seen = set()
    for p in parts:
        c = canonicalize_snippet(p)
        if not c or c in seen:
            continue
        seen.add(c)
        clean_parts.append(c)
    if not clean_parts:
        return ""
    if len(clean_parts) == 1:
        return clean_parts[0]

    # Chinese-only chunks: no extra spaces by default.
    if all(re.fullmatch(r"[\u4e00-\u9fff]+", p) for p in clean_parts):
        return "".join(clean_parts)
    return " ".join(clean_parts)


def normalize_season_episode_candidate(text: str) -> str:
    """
    Normalize common season-episode forms without being overly aggressive.
    """
    compact = re.sub(r"\s+", "", text).strip()

    # SxxExx form.
    if re.search(r"(?i)s\d{1,3}e\d{1,4}", compact):
        return compact.upper()

    # 3x12 form -> S03E12 (more canonical for this project).
    m_x = re.search(r"(?i)\b0*(\d{1,3})x0*(\d{1,4})\b", compact)
    if m_x:
        season = int(m_x.group(1))
        episode = int(m_x.group(2))
        return f"S{season:02d}E{episode:02d}"

    # Range like S01-S22.
    if re.search(r"(?i)s\d{1,3}[-~]s?\d{1,3}", compact):
        return compact.upper()

    return canonicalize_snippet(text)


def normalize_frame_rate_candidate(text: str) -> str:
    cleaned = canonicalize_snippet(text)
    if not cleaned:
        return ""

    fps_match = re.search(r"(?i)\b(\d{2,3}(?:\.\d{1,3})?)\s*fps\b", cleaned)
    if fps_match:
        return f"{fps_match.group(1)}fps"

    num_match = re.fullmatch(r"(\d{2,3}(?:\.\d{1,3})?)", cleaned)
    if num_match:
        return f"{num_match.group(1)}fps"

    return cleaned


def resolve_season_episode_segments(segments: Sequence[str]) -> str:
    """
    Resolve multiple season_episode fragments.
    Priority:
      1) episode-specific formats (SxxExx / 3x12 / 第x季第y集)
      2) season ranges (S01-S22)
      3) season-only (S01 / 第八季)
      4) fallback smart join
    """
    clean_parts: List[str] = []
    seen = set()
    for seg in segments:
        cleaned = canonicalize_snippet(seg)
        if not cleaned or cleaned in seen:
            continue
        seen.add(cleaned)
        clean_parts.append(cleaned)

    if not clean_parts:
        return ""

    if len(clean_parts) == 1:
        return normalize_season_episode_candidate(clean_parts[0])

    # Chinese numerals set for season/episode regex.
    cn_num = r"0-9零一二三四五六七八九十百两兩〇"
    patt_full_episode = re.compile(
        rf"(?i)s\s*\d{{1,3}}\s*e\s*\d{{1,4}}|\b\d{{1,3}}\s*x\s*\d{{1,4}}\b|第\s*[{cn_num}]+\s*季\s*[-_. ]*第?\s*[{cn_num}]+\s*[集话話]?"
    )
    patt_range = re.compile(r"(?i)s\s*\d{1,3}\s*[-~]\s*s?\s*\d{1,3}")
    patt_season_only = re.compile(rf"(?i)\bs\s*\d{{1,3}}\b|第\s*[{cn_num}]+\s*季")

    best: Optional[Tuple[int, int, str]] = None
    # score tuple: (priority, index, candidate) ; higher is better
    for idx, part in enumerate(clean_parts):
        priority = 0
        if patt_full_episode.search(part):
            priority = 3
        elif patt_range.search(part):
            priority = 2
        elif patt_season_only.search(part):
            priority = 1
        else:
            priority = 0

        candidate = normalize_season_episode_candidate(part)
        current = (priority, idx, candidate)
        if best is None or current > best:
            best = current

    if best and best[0] > 0:
        return best[2]

    return smart_join_segments(clean_parts)


def merge_field_segments(parsed_key: str, segments: Sequence[str]) -> str:
    if parsed_key == "season_episode":
        return resolve_season_episode_segments(segments)
    if parsed_key == "frame_rate":
        return normalize_frame_rate_candidate(smart_join_segments(segments))
    return smart_join_segments(segments)


class MediaFilenameNERPredictor:
    """
    Load a fine-tuned token-classification model and parse unknown filenames.
    """

    def __init__(
        self,
        model_dir_or_name: str,
        max_length: int = 192,
        device: Optional[str] = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: torch. Please install with `pip install torch`."
            ) from exc
        try:
            from transformers import (
                AutoConfig,
                AutoModelForTokenClassification,
                AutoTokenizer,
            )
        except ImportError as exc:
            raise ImportError(
                "Missing dependency: transformers. Please install with `pip install transformers`."
            ) from exc

        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir_or_name, use_fast=True)
        model_config = AutoConfig.from_pretrained(model_dir_or_name)
        use_crf = bool(getattr(model_config, "use_crf", False))
        if use_crf:
            self.model = BertCrfForTokenClassification.from_pretrained(
                model_dir_or_name
            )
        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model_dir_or_name
            )
        self.max_length = max_length

        if device is None:
            device = "cuda" if self._torch.cuda.is_available() else "cpu"
        self.device = self._torch.device(device)
        self.model.to(self.device)
        self.model.eval()

    def _id_to_label(self, label_id: int) -> str:
        mapping = self.model.config.id2label
        if isinstance(mapping, dict):
            if label_id in mapping:
                return mapping[label_id]
            if str(label_id) in mapping:
                return mapping[str(label_id)]
        return "O"

    def _decode_token_spans(
        self,
        filename: str,
        pred_ids: Sequence[int],
        offsets: Sequence[Tuple[int, int]],
    ) -> List[Tuple[str, int, int]]:
        entities: List[Tuple[str, int, int]] = []
        cur_entity: Optional[str] = None
        cur_start: Optional[int] = None
        cur_end: Optional[int] = None

        def flush():
            nonlocal cur_entity, cur_start, cur_end
            if (
                cur_entity is not None
                and cur_start is not None
                and cur_end is not None
                and cur_end > cur_start
            ):
                entities.append((cur_entity, cur_start, cur_end))
            cur_entity, cur_start, cur_end = None, None, None

        for pred_id, (start, end) in zip(pred_ids, offsets):
            if start == end:
                flush()
                continue
            if start >= len(filename):
                flush()
                continue

            label = self._id_to_label(int(pred_id))
            if label == "O" or "-" not in label:
                flush()
                continue

            prefix, entity = label.split("-", 1)
            if prefix == "B":
                flush()
                cur_entity, cur_start, cur_end = entity, start, min(end, len(filename))
                continue

            # I-* handling:
            if cur_entity == entity and cur_end is not None and start <= cur_end + 1:
                cur_end = max(cur_end, min(end, len(filename)))
            else:
                # Invalid orphan I-* -> treat as a new B-* segment.
                flush()
                cur_entity, cur_start, cur_end = entity, start, min(end, len(filename))

        flush()
        return entities

    @staticmethod
    def _merge_adjacent_same_entity(
        filename: str,
        spans: Sequence[Tuple[str, int, int]],
    ) -> List[Tuple[str, int, int]]:
        if not spans:
            return []
        spans_sorted = sorted(spans, key=lambda x: (x[1], x[2]))
        merged: List[List[object]] = []

        for entity, start, end in spans_sorted:
            if not merged:
                merged.append([entity, start, end])
                continue
            last_entity, _, last_end = merged[-1]
            gap_text = filename[last_end:start] if start >= last_end else ""
            # Merge same-entity overlap cases first.
            # Example:
            #   (ZH_TITLE, 6, 7) + (ZH_TITLE, 6, 16) -> keep (6, 16)
            same_entity_overlap = entity == last_entity and start <= last_end
            if same_entity_overlap:
                # New span fully contained in previous one: ignore.
                if end <= last_end:
                    continue
                # Partial overlap or same start with longer end: extend.
                merged[-1][2] = max(last_end, end)
                continue

            should_merge = (
                entity == last_entity
                and start >= last_end
                and len(gap_text) <= 4
                and bool(SEPARATOR_ONLY_PATTERN.match(gap_text))
            )
            if should_merge:
                merged[-1][2] = max(last_end, end)
            else:
                merged.append([entity, start, end])

        return [(e, int(s), int(t)) for e, s, t in merged]

    def predict(
        self,
        filename: str,
        return_spans: bool = False,
    ) -> (
        Dict[str, Optional[str]]
        | Tuple[Dict[str, Optional[str]], List[Dict[str, object]]]
    ):
        if not filename:
            empty = {k: None for k in PARSED_KEY_TO_ENTITY}
            return (empty, []) if return_spans else empty

        encoded = self.tokenizer(
            filename,
            return_offsets_mapping=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        encoded = {k: v.to(self.device) for k, v in encoded.items()}

        with self._torch.no_grad():
            logits = self.model(**encoded).logits[0]
        pred_ids = self._torch.argmax(logits, dim=-1).tolist()

        token_spans = self._decode_token_spans(filename, pred_ids, offsets)
        merged_spans = self._merge_adjacent_same_entity(filename, token_spans)

        grouped: Dict[str, List[str]] = defaultdict(list)
        debug_spans: List[Dict[str, object]] = []
        for entity, start, end in merged_spans:
            snippet = canonicalize_snippet(filename[start:end])
            if not snippet:
                continue
            if entity not in ENTITY_TO_PARSED_KEY:
                continue
            parsed_key = ENTITY_TO_PARSED_KEY[entity]
            grouped[parsed_key].append(snippet)
            debug_spans.append(
                {
                    "entity": entity,
                    "parsed_key": parsed_key,
                    "start": start,
                    "end": end,
                    "text": snippet,
                }
            )

        structured = {k: None for k in PARSED_KEY_TO_ENTITY}
        for key, segments in grouped.items():
            merged_text = merge_field_segments(key, segments)
            structured[key] = merged_text if merged_text else None

        if return_spans:
            return structured, debug_spans
        return structured


# ============================================================================
# CLI / Main
# ============================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train NER for media filename parsing."
    )
    parser.add_argument("--dataset_path", type=str, default="parsed_dataset.json")
    parser.add_argument("--output_dir", type=str, default="outputs/media_filename_ner")
    parser.add_argument(
        "--model_name",
        type=str,
        default="xlm-roberta-base",
        help="Recommended: hfl/chinese-roberta-wwm-ext or xlm-roberta-base",
    )
    parser.add_argument("--min_confidence", type=float, default=0.8)
    parser.add_argument("--max_length", type=int, default=192)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument(
        "--model_type",
        type=str,
        default="crf",
        choices=MODEL_TYPE_CHOICES,
        help="NER head type: plain token-classification or CRF-constrained decoding.",
    )
    parser.add_argument(
        "--path_aware_mode",
        type=str,
        default="raw_plus_basename",
        choices=PATH_AWARE_MODES,
        help="Input text composition for train/infer alignment.",
    )
    parser.add_argument(
        "--hard_negative_boost",
        type=int,
        default=2,
        help="Replicate mined hard-negative examples by this factor (1 = disabled).",
    )
    parser.add_argument(
        "--hard_negative_tokens",
        type=str,
        default="2024,1080,1134",
        help="Comma-separated numeric confusion hints used for hard-negative mining.",
    )
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help="Optional cap for quick experiments/debug.",
    )
    parser.add_argument(
        "--demo_filename",
        type=str,
        default="[CancelSub] 卡片戰鬥!! 先導者 overdress 第三季 - 13 [1080P][Baha][WEB-DL][AAC AVC][CHT].mp4",
    )
    return parser.parse_args()


def format_field_stats(field_stats: Dict[str, Dict[str, int]]) -> str:
    lines = []
    for field in PARSED_KEY_TO_ENTITY:
        total = field_stats[field]["total_non_null"]
        matched = field_stats[field]["matched"]
        rate = (matched / total) if total else 0.0
        lines.append(
            f"{field:15s} total={total:6d} matched={matched:6d} rate={rate:.3f}"
        )
    return "\n".join(lines)


def run_training_pipeline(args: argparse.Namespace) -> None:
    try:
        import inspect
        from transformers import DataCollatorForTokenClassification, Trainer, set_seed
    except ImportError as exc:
        raise ImportError(
            "Missing dependency: transformers. Please install with `pip install transformers`."
        ) from exc

    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    set_seed(args.seed)

    LOGGER.info("Loading dataset from %s", dataset_path)
    records = load_json_records(dataset_path)

    hard_negative_tokens = [
        tok.strip() for tok in str(args.hard_negative_tokens).split(",") if tok.strip()
    ]

    LOGGER.info("Preprocessing records and generating BIO labels ...")
    examples, field_stats, global_stats = preprocess_records_to_examples(
        records=records,
        min_confidence=args.min_confidence,
        path_aware_mode=args.path_aware_mode,
        hard_negative_boost=max(1, int(args.hard_negative_boost)),
        hard_negative_tokens=hard_negative_tokens or ("2024", "1080", "1134"),
    )

    if args.max_train_samples is not None and args.max_train_samples > 0:
        rnd = random.Random(args.seed)
        rnd.shuffle(examples)
        examples = examples[: args.max_train_samples]
        LOGGER.info("Using max_train_samples=%d", len(examples))

    if len(examples) < 20:
        raise RuntimeError(
            "Not enough valid examples after preprocessing. "
            f"Got {len(examples)}. Please lower min_confidence or check dataset quality."
        )

    LOGGER.info(
        "Preprocess stats: input=%d kept=%d low_conf=%d bad_record=%d hard_neg_candidates=%d hard_neg_added=%d",
        global_stats["input_records"],
        global_stats["kept_records"],
        global_stats["skipped_low_conf"],
        global_stats["skipped_bad_record"],
        global_stats["hard_negative_candidates"],
        global_stats["hard_negative_added"],
    )
    LOGGER.info(
        "Training settings: model_type=%s path_aware_mode=%s hard_negative_boost=%d hard_negative_tokens=%s",
        args.model_type,
        args.path_aware_mode,
        args.hard_negative_boost,
        hard_negative_tokens,
    )
    LOGGER.info("Field-level alignment stats:\n%s", format_field_stats(field_stats))

    LOGGER.info("Building train/validation/test splits ...")
    dataset_splits = build_dataset_splits(examples, seed=args.seed)
    LOGGER.info(
        "Split sizes: train=%d validation=%d test=%d",
        len(dataset_splits["train"]),
        len(dataset_splits["validation"]),
        len(dataset_splits["test"]),
    )

    LOGGER.info("Initializing labels, tokenizer, and model ...")
    _, label2id, id2label = build_label_mappings()
    tokenizer, model = initialize_tokenizer_and_model(
        model_name=args.model_name,
        label2id=label2id,
        id2label=id2label,
        model_type=args.model_type,
    )

    LOGGER.info("Tokenizing and aligning token labels ...")
    tokenized_splits = dataset_splits.map(
        tokenize_and_align_labels,
        batched=True,
        fn_kwargs={
            "tokenizer": tokenizer,
            "label2id": label2id,
            "max_length": args.max_length,
        },
        remove_columns=dataset_splits["train"].column_names,
        desc="Tokenize + Align labels",
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    compute_metrics = build_compute_metrics_fn(id2label)
    training_args = build_training_arguments(args)

    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": tokenized_splits["train"],
        "eval_dataset": tokenized_splits["validation"],
        "data_collator": data_collator,
        "compute_metrics": compute_metrics,
    }
    trainer_init_params = set(inspect.signature(Trainer.__init__).parameters.keys())
    if "processing_class" in trainer_init_params:
        trainer_kwargs["processing_class"] = tokenizer
    else:
        trainer_kwargs["tokenizer"] = tokenizer

    trainer = Trainer(**trainer_kwargs)

    LOGGER.info("Starting training ...")
    trainer.train()

    LOGGER.info("Evaluating on validation split ...")
    val_metrics = trainer.evaluate(tokenized_splits["validation"])
    LOGGER.info("Validation metrics: %s", val_metrics)

    LOGGER.info("Evaluating on test split ...")
    test_metrics = trainer.evaluate(tokenized_splits["test"], metric_key_prefix="test")
    LOGGER.info("Test metrics: %s", test_metrics)

    LOGGER.info("Saving model + tokenizer to %s", output_dir)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))

    LOGGER.info("Running single-sample inference demo ...")
    predictor = MediaFilenameNERPredictor(
        model_dir_or_name=str(output_dir),
        max_length=args.max_length,
    )
    prediction, spans = predictor.predict(args.demo_filename, return_spans=True)
    print("\n=== Demo Inference ===")
    print(f"Input filename: {args.demo_filename}")
    print("Predicted structured dict:")
    print(json.dumps(prediction, ensure_ascii=False, indent=2))
    print("Predicted entity spans:")
    print(json.dumps(spans, ensure_ascii=False, indent=2))


def setup_logging() -> None:
    setup_shared_logging("INFO")


if __name__ == "__main__":
    setup_logging()
    cli_args = parse_args()
    try:
        run_training_pipeline(cli_args)
    except KeyboardInterrupt:
        LOGGER.warning("Interrupted by user.")
    except Exception as exc:  # pylint: disable=broad-except
        LOGGER.exception("Pipeline failed: %s", exc)
        raise
