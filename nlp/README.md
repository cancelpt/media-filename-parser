# NLP Pipelines

[简体中文](README.zh-CN.md)

This directory contains optional NLP training, evaluation, and inference workflows for media filename parsing.

For package installation and rule-based integration, see the root [README](../README.md).

## Structure

- `ner/`: BIO tagging pipeline (`plain` and `BERT+CRF`) and evaluation tools
- `qwen/`: Qwen LoRA SFT pipeline and inference/evaluation tools
- `shared.py`: shared helpers (including path-aware input composition)

## Core Scripts

- `ner/train_media_filename_ner.py`
- `ner/predict.py`
- `ner/generate_hard_negative_feed.py`
- `ner/evaluate_model_plots.py`
- `ner/compare_low_confidence_with_model.py`
- `ner/plot_scores.py`
- `qwen/evaluate_qwen_sft_parser.py`
- `qwen/interactive_qwen_sft.py`
- `qwen/qwen_sft_parser/data_prep.py`
- `qwen/qwen_sft_parser/train.py`
- `qwen/qwen_sft_parser/inference.py`

## Path-Aware Input Modes

Both NER and Qwen workflows support:

- `basename`
- `raw_path`
- `raw_plus_basename` (default/recommended)

## Quick Commands

```bash
python -m nlp.ner.generate_hard_negative_feed --help
python -m nlp.ner.train_media_filename_ner --help
python -m nlp.ner.predict --help
python -m nlp.ner.evaluate_model_plots --help
python -m nlp.qwen.evaluate_qwen_sft_parser --help
python -m nlp.qwen.interactive_qwen_sft --help
python -m nlp.qwen.qwen_sft_parser.data_prep --help
python -m nlp.qwen.qwen_sft_parser.train --help
python -m nlp.qwen.qwen_sft_parser.inference --help
```

## Example Workflow (NER)

1. Generate hard-negative data:

```bash
python -m nlp.ner.generate_hard_negative_feed \
  --input_json parsed_dataset.json \
  --output_json data/ner_hard_negative_feed.json \
  --repeat_factor 2 \
  --mode append \
  --path_aware_mode raw_plus_basename
```

2. Train:

```bash
python -m nlp.ner.train_media_filename_ner \
  --dataset_path data/ner_hard_negative_feed.json \
  --output_dir outputs/media_filename_ner_crf \
  --model_name xlm-roberta-base \
  --model_type crf \
  --path_aware_mode raw_plus_basename
```

3. Predict:

```bash
python -m nlp.ner.predict \
  --model_dir outputs/media_filename_ner_crf \
  --text "[CancelSub] Cardfight Vanguard overDress S03E13 [1080P][WEB-DL][AAC AVC].mp4" \
  --path_aware_mode raw_plus_basename
```

## Example Workflow (Qwen LoRA SFT)

1. Prepare training data:

```bash
python -m nlp.qwen.qwen_sft_parser.data_prep \
  --input_json parsed_dataset.json \
  --output_dir data/qwen_sft_parser \
  --train_ratio 0.9 \
  --seed 42 \
  --path_aware_mode raw_plus_basename
```

2. Train:

```bash
python -m nlp.qwen.qwen_sft_parser.train \
  --base_model Qwen/Qwen3-1.7B-Base \
  --dataset_dir data/qwen_sft_parser \
  --output_root outputs/qwen_sft_parser \
  --num_train_epochs 3 \
  --report_to none
```

3. Inference:

```bash
python -m nlp.qwen.qwen_sft_parser.inference \
  --base_model Qwen/Qwen3-1.7B-Base \
  --adapter_dir outputs/qwen_sft_parser/<run_dir>/adapter \
  --text "[CancelSub] Cardfight Vanguard overDress S03E13 [1080P][WEB-DL][AAC AVC].mp4" \
  --path_aware_mode raw_plus_basename \
  --merge_lora
```

## Notes

- Input/output paths are generally relative to repository root.
- Training outputs and checkpoints are intentionally ignored by `.gitignore`.
