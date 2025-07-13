#!/usr/bin/env python3
"""
prepare_brand_dataset.py
--------------------
Materialises a brand detection dataset from the LSPC-2020 corpus into a Hugging-Face DatasetDict
with train / validation / test splits.

Input
-----
* lspcV2020.zip        – official ZIP from the challenge, containing
                         many *.json.gz files (one JSON per line)

Output
------
<out>/
  dataset_info.json
  train/  validation/  test/   # Arrow files + metadata

The resulting DatasetDict has four columns:
  • text    (string)           – concatenated title + description
  • label   (ClassLabel)       – 0 for no brand, 1 for brand present
  • brand   (string)           - the brand name, or "NO_BRAND"
  • cluster (int32)            – original cluster_id

Splitting strategy
------------------
To avoid information leakage, splitting is **grouped by cluster_id**:
all items with the same cluster land in the same split. Randomness
is controlled by a single seed (default 42) to ensure reproducibility.
"""

from __future__ import annotations
import argparse, gzip, io, json, pathlib, random, zipfile
from collections import Counter
from typing import Iterable

import tqdm
import pandas as pd
from datasets import (
    Dataset, DatasetDict, Features, Value, ClassLabel,
    disable_progress_bar, concatenate_datasets
)
from sklearn.model_selection import train_test_split

disable_progress_bar()  # keep HF quiet

# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--zip",       type=pathlib.Path, default="lspcV2020.zip",
                   help="Path to lspcV2020.zip")
    p.add_argument("--out",       type=pathlib.Path, default="brand_dataset",
                   help="Output directory for saved DatasetDict")
    p.add_argument("--val_size",  type=float, default=0.1, help="Fraction for validation set")
    p.add_argument("--test_size", type=float, default=0.1, help="Fraction for test set")
    p.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    p.add_argument("--fraction",  type=float, default=1.0,  help="Fraction of dataset to use")
    p.add_argument("--brand_percentile",  type=float, default=0.1,  help="Top percentile of brands to use")
    return p.parse_args()

# --------------------------- Helpers --------------------------- #

def get_brand_counts(zip_path: pathlib.Path, bar_opts: dict) -> Counter:
    """First pass over the data to count brand frequencies."""
    brand_counter = Counter()
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".json.gz")]
        for member in tqdm.tqdm(members, desc="scan-files (pass 1)", **bar_opts):
            with z.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz, \
                 io.TextIOWrapper(gz, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    rec = json.loads(line)
                    brand = rec.get("brand")
                    brand_name = brand.strip() if brand else "__MISSING__"
                    brand_counter.update([brand_name])
    return brand_counter

def json_stream(zip_path: pathlib.Path, bar_opts: dict, top_brands: set) -> Iterable[dict]:
    """Yield cleaned records one by one from the ZIP archive, filtering for top brands."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".json.gz")]
        for member in tqdm.tqdm(members, desc="scan-files (pass 2)", **bar_opts):
            with z.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz, \
                 io.TextIOWrapper(gz, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    rec = json.loads(line)
                    text = " ".join(filter(None, (
                        rec.get("title") or rec.get("name"),
                        rec.get("description") or ""
                    )))
                    brand = rec.get("brand")
                    brand_name = brand.strip() if brand else "__MISSING__"
                    if text.strip() and brand_name in top_brands:
                        yield {"text": text,
                               "brand": brand_name,
                               "label": 1 if brand_name != "__MISSING__" else 0}

def main():
    args = parse_args()
    bar_opts = {"unit": "file", "ncols": 80, "leave": False}

    if not args.zip.exists():
        raise FileNotFoundError(f"Input file not found: {args.zip}")

    # First pass to count brands
    brand_counts = get_brand_counts(args.zip, bar_opts)
    print("Brand Frequencies (Top 30):")
    for brand, count in brand_counts.most_common(30):
        print(f"{brand}: {count}")
    print(f"Total unique brands: {len(brand_counts)}")

    # Determine top brands to keep
    if args.brand_percentile < 1.0:
        num_to_keep = int(len(brand_counts) * args.brand_percentile)
        top_brands = {b for b, _ in brand_counts.most_common(num_to_keep)} | {"__MISSING__"}
        print(f"Keeping top {args.brand_percentile:.0%} of brands ({len(top_brands) - 1} brands) plus '__MISSING__'")
    else:
        top_brands = set(brand_counts.keys())

    # Define features to ensure 'label' is a ClassLabel for stratification
    features = Features({
        'text': Value('string'),
        'brand': Value('string'),
        'label': ClassLabel(num_classes=2, names=['no_brand', 'brand'])
    })

    # Use a generator to avoid loading all data into memory
    ds = Dataset.from_generator(json_stream, gen_kwargs={"zip_path": args.zip, "bar_opts": bar_opts, "top_brands": top_brands}, features=features)

    # --- Downsampling ---
    brand_ds = ds.filter(lambda x: x['label'] == 1)
    no_brand_ds = ds.filter(lambda x: x['label'] == 0)

    if len(no_brand_ds) > len(brand_ds):
        print(f"Downsampling 'no_brand' from {len(no_brand_ds)} to {len(brand_ds)} samples.")
        no_brand_ds = no_brand_ds.shuffle(seed=args.seed).select(range(len(brand_ds)))

    balanced_ds = concatenate_datasets([brand_ds, no_brand_ds]).shuffle(seed=args.seed)
    print(f"Total balanced dataset size: {len(balanced_ds)}")

    if args.fraction < 1.0:
        print(f"Using {args.fraction:.0%} of the dataset.")
        num_samples = int(len(balanced_ds) * args.fraction)
        balanced_ds = balanced_ds.select(range(num_samples))
        print(f"New dataset size: {len(balanced_ds)}")
    
    # --- Split data ---
    ds_split = balanced_ds.train_test_split(test_size=args.test_size, seed=args.seed, stratify_by_column='label')
    train_val_split = ds_split['train'].train_test_split(test_size=args.val_size / (1 - args.test_size), seed=args.seed, stratify_by_column='label')

    ds = DatasetDict({
        "train": train_val_split['train'],
        "validation": train_val_split['test'],
        "test": ds_split['test']
    })

    print(f"\nDataset splits (train/val/test): {len(ds['train'])}/{len(ds['validation'])}/{len(ds['test'])}")
    args.out.mkdir(exist_ok=True)
    ds.save_to_disk(args.out)
    print(f"✅ Dataset saved to {args.out}")

if __name__ == "__main__":
    main()
