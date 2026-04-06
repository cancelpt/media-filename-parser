# NLP Pipelines

[English](README.md)

该目录包含媒体文件名解析相关的可选 NLP 流水线：训练、评估与推理。

如果你只关心安装与规则解析集成，请先看根目录 [README.zh-CN.md](../README.zh-CN.md)。

## 目录结构

- `ner/`：BIO 标注的 NER 流水线（`plain` 与 `BERT+CRF`）及评估工具
- `qwen/`：Qwen LoRA SFT 流水线及推理/评估工具
- `shared.py`：共享工具（含 path-aware 输入拼接）

## 核心脚本

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

## Path-Aware 输入模式

NER 与 Qwen 都支持：

- `basename`
- `raw_path`
- `raw_plus_basename`（默认值）

## 快速命令

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

## 示例流程（NER）

1. 生成 hard-negative 数据：

```bash
python -m nlp.ner.generate_hard_negative_feed \
  --input_json parsed_dataset.json \
  --output_json data/ner_hard_negative_feed.json \
  --repeat_factor 2 \
  --mode append \
  --path_aware_mode raw_plus_basename
```

2. 训练：

```bash
python -m nlp.ner.train_media_filename_ner \
  --dataset_path data/ner_hard_negative_feed.json \
  --output_dir outputs/media_filename_ner_crf \
  --model_name xlm-roberta-base \
  --model_type crf \
  --path_aware_mode raw_plus_basename
```

3. 推理：

```bash
python -m nlp.ner.predict \
  --model_dir outputs/media_filename_ner_crf \
  --text "[CancelSub] Cardfight Vanguard overDress S03E13 [1080P][WEB-DL][AAC AVC].mp4" \
  --path_aware_mode raw_plus_basename
```

## 示例流程（Qwen LoRA SFT）

1. 准备训练数据：

```bash
python -m nlp.qwen.qwen_sft_parser.data_prep \
  --input_json parsed_dataset.json \
  --output_dir data/qwen_sft_parser \
  --train_ratio 0.9 \
  --seed 42 \
  --path_aware_mode raw_plus_basename
```

2. 训练：

```bash
python -m nlp.qwen.qwen_sft_parser.train \
  --base_model Qwen/Qwen3-1.7B-Base \
  --dataset_dir data/qwen_sft_parser \
  --output_root outputs/qwen_sft_parser \
  --num_train_epochs 3 \
  --report_to none
```

3. 推理：

```bash
python -m nlp.qwen.qwen_sft_parser.inference \
  --base_model Qwen/Qwen3-1.7B-Base \
  --adapter_dir outputs/qwen_sft_parser/<run_dir>/adapter \
  --text "[CancelSub] Cardfight Vanguard overDress S03E13 [1080P][WEB-DL][AAC AVC].mp4" \
  --path_aware_mode raw_plus_basename \
  --merge_lora
```

## 说明

- 输入/输出路径通常相对仓库根目录。
- 训练输出与 checkpoint 已在 `.gitignore` 中忽略。
