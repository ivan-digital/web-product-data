#!/usr/bin/env python3
"""
Measure inference latency for a Qwen/Qwen3 model fine-tuned with LoRA.

The script loads a *merged* checkpoint produced by your training run,
feeds **10 000** product texts through the model, and reports:
    • mean / median latency per example
    • 95th & 99th percentile latencies
    • total throughput (examples / second)

The product texts can be supplied either as:
    1. **Dataset** saved via `datasets.save_to_disk` (the same format your
       training script uses) – specify with `--dataset`.
    2. **Plain text file** – one product description per line – specify
       with `--txt`.

Example
-------
python measure_latency_qwen3.py \
    --model ./qwen3_lora_prodcat_adam/merged \
    --dataset ./lspc_dataset \
    --split test \
    --batch 64

A short warm-up run is performed before timing begins to let GPUs reach
steady performance.
"""
from __future__ import annotations
import argparse, os, time, json, math, warnings
from typing import Sequence

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_from_disk, Dataset
from tqdm.auto import tqdm

# Also suppress HuggingFace transformers warnings
import logging
logging.getLogger("transformers.modeling_utils").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
import os
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

# ────────────────────────────────────────────────────────────────────────────────

def infer_device_dtype() -> tuple[torch.device, torch.dtype]:
    """Choose the best available device and dtype."""
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32

# ────────────────────────────────────────────────────────────────────────────────

def load_products(args) -> Sequence[str]:
    """Load **at most** `args.limit` product texts from dataset or txt file."""
    if args.dataset:
        ds = load_from_disk(args.dataset)
        split = args.split or "test"
        if split not in ds:
            raise ValueError(f"Split '{split}' not found in dataset; available: {list(ds.keys())}")
        texts = ds[split][: args.limit]["text"]
    elif args.txt:
        with open(args.txt, "r", encoding="utf-8") as fh:
            texts = [line.rstrip("\n") for line in fh]
        texts = texts[: args.limit]
    else:
        raise ValueError("Either --dataset or --txt must be given.")

    if len(texts) < args.limit:
        warnings.warn(f"Only found {len(texts)} items – less than requested {args.limit}.")
    return texts

# ────────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Qwen3 inference latency benchmark")

    grp_model = ap.add_argument_group("Model")
    grp_model.add_argument("--model", type=str, default="./category_classification/qwen3_lora_product_adam/merged", help="Path to merged PEFT model directory")

    grp_data = ap.add_argument_group("Data source (choose one)")
    grp_data.add_argument("--dataset", default="./category_classification/lspc_dataset", type=str, help="Path to dataset saved with datasets.save_to_disk")
    grp_data.add_argument("--split", type=str, default="test", help="Dataset split to use (default: test)")
    grp_data.add_argument("--txt", type=str, help="Plain-text file – one product per line")

    grp = ap.add_argument_group("Runtime")
    grp.add_argument("--limit", type=int, default=10_000, help="Number of products to process (default: 10 000)")
    grp.add_argument("--batch", type=int, default=32, help="Batch size (per forward pass)")
    grp.add_argument("--seq_len", type=int, default=256, help="Max sequence length (tokens)")
    grp.add_argument("--warmup_batches", type=int, default=5, help="Batches to run before timing starts")

    args = ap.parse_args()

    # Select device
    device, dtype = infer_device_dtype()
    print(f"🖥️  Using device: {device} – dtype: {dtype}")

    # Load tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, trust_remote_code=True)

    # Ensure proper pad token
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token or "<|pad|>"
    tok.padding_side = "right"
    
    # Set model's pad token ID to match tokenizer
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tok.pad_token_id

    model.to(device, dtype=dtype)
    model.eval()

    # Report model parameters and precision
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    param_dtype = next(model.parameters()).dtype
    print(f"🔢 Model parameters: {total_params:,} total, {trainable_params:,} trainable")
    print(f"🎯 Parameter precision: {param_dtype}")
    print(f"📦 Batch size: {args.batch}")

    # Load products
    products = load_products(args)
    n_items = len(products)
    print(f"📦 Loaded {n_items} product texts")

    # Helper: batch iterator
    def batch_iter(seq: Sequence[str], batch_size: int):
        for i in range(0, len(seq), batch_size):
            yield seq[i : i + batch_size]

    latencies = []  # seconds per item

    # Warm-up
    if args.warmup_batches > 0:
        warm_texts = products[: args.warmup_batches * args.batch]
        for batch in batch_iter(warm_texts, args.batch):
            with torch.no_grad():
                toks = tok(batch, max_length=args.seq_len, padding=True, truncation=True, return_tensors="pt").to(device)
                _ = model(**toks)
        torch.cuda.empty_cache() if device.type == "cuda" else None
        print(f"🔥 Warm-up: {args.warmup_batches} batches done")

    # Timed run
    start_time = time.perf_counter()
    with torch.no_grad():
        for batch in tqdm(batch_iter(products, args.batch), total=math.ceil(n_items / args.batch), desc="Inferencing", unit="batch"):
            t0 = time.perf_counter()
            toks = tok(batch, max_length=args.seq_len, padding=True, truncation=True, return_tensors="pt").to(device)
            _ = model(**toks)
            torch.cuda.synchronize() if device.type == "cuda" else None
            t1 = time.perf_counter()
            latencies.append((t1 - t0) / len(batch))
    total_time = time.perf_counter() - start_time

    # Stats
    lat = np.array(latencies)
    stats = {
        "mean_ms": lat.mean() * 1000,
        "median_ms": np.median(lat) * 1000,
        "p95_ms": np.percentile(lat, 95) * 1000,
        "p99_ms": np.percentile(lat, 99) * 1000,
        "throughput_eps": n_items / total_time,
    }

    print("\n🕒 Latency summary:")
    for k, v in stats.items():
        if k.endswith("_ms"):
            print(f"  {k:<12}: {v:,.2f}  ms")
        else:
            print(f"  {k:<12}: {v:,.2f}  samples/s")


if __name__ == "__main__":
    main()
