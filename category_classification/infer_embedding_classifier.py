#!/usr/bin/env python3
"""
Simple inference script for the EmbeddingClassifier checkpoints.

Usage:
    poetry run python category_classification/infer_embedding_classifier.py \
      --data ./lspc_dataset_full \
      --checkpoint ./embedding_classifier_prodcat/checkpoint-epoch5.pt \
      --split test \
      --batch_size 256

Outputs:
    - Prints macro/micro F1 and accuracy for the requested split.
    - Writes predictions/labels to <checkpoint_dir>/predictions-<split>.jsonl (optional via --save_preds).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List
import time
import numpy as np
import torch
from datasets import load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure project root importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from category_classification.embedding_classifier import (
    EmbeddingClassifier,
    EmbeddingClassifierConfig,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run inference using a saved EmbeddingClassifier checkpoint.")
    p.add_argument("--data", type=str, required=True, help="Path to HF DatasetDict (train/validation/test).")
    p.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint-epochX.pt.")
    p.add_argument("--split", type=str, default="test", choices=["train", "validation", "test"], help="Dataset split.")
    p.add_argument("--batch_size", type=int, default=256, help="Per-device batch size for inference.")
    p.add_argument("--max_length", type=int, default=256, help="Max token length.")
    p.add_argument("--save_preds", action="store_true", help="Save predictions+labels to JSONL next to checkpoint.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds = load_from_disk(args.data)
    if args.split not in ds:
        raise ValueError(f"Split '{args.split}' not found in dataset.")
    ds_split = ds[args.split]
    num_labels = ds["train"].features["label"].num_classes

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    tokenizer.padding_side = "right"

    def tokenize(batch):
        enc = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_length,
        )
        enc["labels"] = batch["label"]
        return enc

    cols_to_remove = [c for c in ds_split.column_names if c not in {"text", "label"}]
    ds_tok = ds_split.map(tokenize, batched=True, remove_columns=cols_to_remove)
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])

    loader = DataLoader(ds_tok, batch_size=args.batch_size, shuffle=False)

    ckpt_path = Path(args.checkpoint)
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    config = EmbeddingClassifierConfig(
        vocab_size=len(tokenizer),
        num_labels=num_labels,
        pad_token_id=tokenizer.pad_token_id,
        dropout=0.1,
    )
    model = EmbeddingClassifier(config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    all_preds: List[int] = []
    all_labels: List[int] = []
    batch_times: List[float] = []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            t0 = time.time()
            out = model(**batch)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            batch_times.append(time.time() - t0)
            preds = torch.argmax(out["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    metrics = {
        "macro_f1": float(f1_score(labels_np, preds_np, average="macro")),
        "micro_f1": float(f1_score(labels_np, preds_np, average="micro")),
        "accuracy": float(accuracy_score(labels_np, preds_np)),
    }
    print(f"{args.split} metrics: {metrics}")

    bt = np.array(batch_times, dtype=np.float64)
    per_example_ms = bt * 1000.0 / args.batch_size
    latency = {
        "batch_mean_s": float(bt.mean()),
        "batch_p50_s": float(np.percentile(bt, 50)),
        "batch_p95_s": float(np.percentile(bt, 95)),
        "batch_p99_s": float(np.percentile(bt, 99)),
        "example_mean_ms": float(per_example_ms.mean()),
        "example_p95_ms": float(np.percentile(per_example_ms, 95)),
        "example_p99_ms": float(np.percentile(per_example_ms, 99)),
    }
    print(f"{args.split} latency: {latency}")

    if args.save_preds:
        out_dir = ckpt_path.parent
        out_file = out_dir / f"predictions-{args.split}.jsonl"
        with open(out_file, "w", encoding="utf-8") as fh:
            for lbl, pred in zip(all_labels, all_preds):
                fh.write(json.dumps({"label": lbl, "pred": pred}) + "\n")
        print(f"Saved predictions to {out_file}")


if __name__ == "__main__":
    main()
