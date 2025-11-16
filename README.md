# Product Title Classification based on Web Product Common Crawl subset

Fine-tuned language models for product title classification using the Large-Scale Product Corpus (LSPC) V2020 dataset. The project focuses on **6-class category classification** using LoRA fine-tuning of Qwen embedding models.

## Project Structure

```
web-product-data/
├── category_classification/           # Product title classification components
│   ├── build_lspc_dataset.py         # Dataset preparation script
│   ├── build_lspc_dataset_full.py    # Dataset with extra full split
│   ├── train_qwen3_lora.py           # Main training script
│   ├── train_qwen3_lora_v2.py        # Alternative training script
│   ├── train_embedding_classifier.py # Pure PyTorch classifier trainer
│   ├── embedding_classifier.py       # Pure PyTorch embedding encoder
│   └── results_category.txt          # Training results and metrics
├── docs/
│   └── embedding_classifier_architecture.md  # Detailed model description
├── webdata_discovery.py              # Data exploration tool
└── pyproject.toml                    # Dependencies
```

## Product Title Classification

### Overview

The task classifies product titles into 6 main categories based on web product data from Common Crawl:

0. **Automotive** - Car parts, accessories, tools
1. **Baby** - Baby products, toys, care items
2. **Books** - All types of books and publications
3. **Clothing** - Apparel, fashion items
4. **Jewelry** - Jewelry, watches, accessories
5. **Shoes** - Footwear of all types

### Quick Start

1. **Prepare Dataset (train/val/test + full):**
```bash
python category_classification/build_lspc_dataset.py --zip lspcV2020.zip --out lspc_dataset
# Or include a `full` split:
python category_classification/build_lspc_dataset_full.py --zip lspcV2020.zip --out lspc_dataset_full
# Append --add_language to annotate records with detected language codes (requires ftlangdetect or langdetect)
```

2. **Train Model:**
```bash
python category_classification/train_qwen3_lora.py --data ./lspc_dataset --out ./qwen3_lora_prodcat --batch 128
```

3. **Train Pure PyTorch Embedding Classifier (optional – push batch size/grad_accum to fill VRAM):**
```bash
python category_classification/train_embedding_classifier.py \
    --data ./lspc_dataset \
    --out  ./embedding_classifier_prodcat
```

### Best Results

**Test Performance:**
- **Macro F1**: 0.8360 (83.60%)
- **Accuracy**: 0.8791 (87.91%)

**Best Configuration:**
- Optimizer: adamw_torch  
- Learning Rate: 5e-5
- LoRA: r=16, alpha=32
- Epochs: 1

### Pure PyTorch Embedding Classifier

`category_classification/embedding_classifier.py` contains a compact, encoder-only
(bidirectional self-attention) architecture implemented directly with PyTorch,
so tokens can attend left/right (no causal mask).  Instantiate it with the
tokenizer’s vocab size and your target label count:

```python
from category_classification.embedding_classifier import (
    EmbeddingClassifierConfig,
    EmbeddingClassifier,
)

config = EmbeddingClassifierConfig(
    vocab_size=len(tokenizer),  # always match the tokenizer vocabulary size
    num_labels=6,
)
model = EmbeddingClassifier(config)
outputs = model(**batch, return_embeddings=True)
pooled = outputs["pooled_embeddings"]
tokens = outputs["token_embeddings"]
```

Always determine `vocab_size` from the tokenizer you plan to use (`len(tokenizer)`
for HuggingFace tokenizers) so the embedding matrix and token IDs stay aligned.

The forward pass accepts `input_ids`, `attention_mask`, and optional `labels`,
returning a dictionary with `logits` (and `loss` during training).  Set
`return_embeddings=True` to access both CLS-style pooled embeddings and the
final per-token representations.  You can plug the module into a custom PyTorch
training loop or wrap it with your preferred trainer/optimizer stack.

### Training the Embedding Classifier

`category_classification/train_embedding_classifier.py` reproduces the category
classification experiment end-to-end using the pure PyTorch model.  It loads the
LSPC dataset from disk, tokenizes with the Qwen tokenizer, and optimizes the
EmbeddingClassifier using AdamW.  The default configuration (10 layers, 384-dim
hidden size, 6/2 attention heads, 1536-dim feed-forward) contains roughly
**80 million parameters** (it prints the exact count on startup).

```bash
python category_classification/train_embedding_classifier.py \
    --data ./lspc_dataset \
    --out  ./embedding_classifier_prodcat \
    --epochs 3 --batch_size 64 --grad_accum 2 --save_every 1 --amp --memory_summary_freq 5000
```

Metrics for validation/test are written to `metrics.json`, and the trained
weights + tokenizer are saved under the output directory. Every `--save_every`
epochs (default: 1) a `checkpoint-epochX.pt` snapshot is written, so you can
resume later with `--resume path/to/checkpoint-epochX.pt`. Language metrics are
enabled by default (requires `ftlangdetect` or `langdetect`); use
`--no_lang_metrics` to disable them and `--lang_min_samples` / `--lang_top_k` to
control which languages are reported. Add `--amp` to enable CUDA mixed precision
(AMP) for larger batches and faster training. Use `--grad_accum` to keep an
effective batch size near 1–2k tokens without running out of memory, and
optionally set `--memory_summary_freq` for periodic allocator stats.

### Full Corpus Split

If you want to train on **all** LSPC examples while still keeping clean
evaluation splits, build the dataset with
`category_classification/build_lspc_dataset_full.py`. It behaves like the
original builder but adds a `"full"` split (configurable via
`--full_split_name`). You can point your trainer at `train`/`validation`/`test`
for benchmarking and use the `full` split for large-scale pretraining or
distillation runs.

## Installation

### Using Poetry (Recommended)

This project uses [Poetry](https://python-poetry.org/) for dependency management. Poetry automatically handles virtual environments and ensures reproducible builds.

1. **Install Poetry** (if not already installed):
```bash
# On Windows (PowerShell)
(Invoke-WebRequest -Uri https://install.python-poetry.org -UseBasicParsing).Content | py -

# On macOS/Linux
curl -sSL https://install.python-poetry.org | python3 -
```

2. **Install dependencies:**
```bash
poetry install
```

3. **Activate the virtual environment:**
```bash
poetry shell
```

### Alternative: pip install

If you prefer using pip directly:
```bash
pip install datasets transformers peft scikit-learn torch torchvision torchaudio accelerate bitsandbytes tqdm numpy
```

### Required Files
- **lspcV2020.zip**: Download from [Web Data Commons](https://webdatacommons.org/largescaleproductcorpus/v2020/)
- **PDC2020_map.tsv**: Auto-generated during dataset preparation

## Hardware Requirements

- **GPU**: 8GB+ VRAM recommended
- **RAM**: 16GB+ system memory  
- **Storage**: 50GB+ free space
