# Product Title Classification based on Web Product Common Crawl subset

Fine-tuned language models for product title classification using the Large-Scale Product Corpus (LSPC) V2020 dataset. The project focuses on **6-class category classification** using LoRA fine-tuning of Qwen embedding models.

## Project Structure

```
web-product-data/
├── category_classification/           # Product title classification components
│   ├── build_lspc_dataset.py         # Dataset preparation script
│   ├── train_qwen3_lora.py           # Main training script
│   ├── train_qwen3_lora_v2.py        # Alternative training script
│   └── results_category.txt          # Training results and metrics
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

1. **Prepare Dataset:**
```bash
python category_classification/build_lspc_dataset.py --zip lspcV2020.zip --out lspc_dataset
```

2. **Train Model:**
```bash
python category_classification/train_qwen3_lora.py --data ./lspc_dataset --out ./qwen3_lora_prodcat --batch 128
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
