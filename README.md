# MergeEDA v2

Multimodal VLM pipeline for fine-tuning Qwen3-VL on AMBA specification documents.

## Project Structure

```
MergeEDA_v2/
├── configs/
│   ├── augmentation/amba_eval_qa.yaml
│   ├── evaluation/
│   │   ├── eval_base_model.yaml
│   │   └── model/{qwen_vl_model,qwen_vl_finetuned_model}.yaml
│   └── preprocess/{amba_document,generate_train_data,merge_train_data}.yaml
├── data/datasets/amba_document/
│   ├── raw/                    # Input AMBA PDF files
│   ├── processed/<spec>/
│   │   ├── chunks/             # Markdown chunks (1.md, 2.md, ...)
│   │   └── materials/          # Extracted images (.jpg) and tables (.txt)
│   ├── eval_qa/<spec>/         # QA JSON files (*_concept/material/reasoning.json)
│   └── sft_data/               # SFT training JSON files
├── scripts/
│   ├── preprocess/{amba_document,generate_train_data,merge_train_data}.py
│   ├── augmentation/amba_eval_qa.py
│   ├── evaluation/eval_base_model.py
│   ├── train/train.sh
│   └── inference.sh
├── src/mergeeda/
│   ├── preprocess/{OCRParser,TrainDataGenerator,TrainDataMerger}.py
│   ├── augmentation/{QGenerator,EvalQSetGenerator}.py
│   ├── evaluation/{AnswerGenerator,LLMJudgeEvaluator}.py
│   └── models/{builder,qwen_vl_model,qwen_vl_finetuned_model}.py
├── qwen-vl-finetune/           # Qwen-VL LoRA fine-tuning submodule
└── pyproject.toml
```

## Setup

```bash
pip install -e .
export OPENAI_API_KEY="..."
```

## Pipeline

### 1. Parse PDF (OCR)

```bash
python scripts/preprocess/amba_document.py \
  input_pdf="data/datasets/amba_document/raw/<file>.pdf" \
  output_dir="data/datasets/amba_document/processed/<spec>"
```

### 2. Generate Eval QA Set

```bash
python scripts/augmentation/amba_eval_qa.py \
  chunks_dir="data/datasets/amba_document/processed/<spec>/chunks" \
  materials_dir="data/datasets/amba_document/processed/<spec>/materials" \
  output_dir="data/datasets/amba_document/eval_qa/<spec>" \
  output_name="<name>"
```

### 3. Generate SFT Training Data

```bash
python scripts/preprocess/generate_train_data.py \
  "train_files=[data/datasets/amba_document/eval_qa/<spec>/<name>_train.json]"
```

### 4. Merge SFT Data

```bash
python scripts/preprocess/merge_train_data.py
# → data/datasets/amba_document/sft_data/merged_train.json
```

### 5. Fine-tune (LoRA)

```bash
bash scripts/train/train.sh Qwen/Qwen3-VL-4B-Instruct [lora_rank]
# Outputs: outputs/<slug>-lora-<rank>-amba-<timestamp>/
```

### 6. Evaluate

```bash
# Direct
python scripts/evaluation/eval_base_model.py \
  qa_dir="data/datasets/amba_document/eval_qa/<spec>" \
  materials_dir="data/datasets/amba_document/processed/<spec>/materials" \
  chunks_dir="data/datasets/amba_document/processed/<spec>/chunks" \
  output_dir="data/evaluation/base_model/<spec>"
# → preds.json, scores.json
```
