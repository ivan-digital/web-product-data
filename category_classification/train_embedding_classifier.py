#!/usr/bin/env python3
"""
Train the pure PyTorch EmbeddingClassifier on the LSPC dataset.

The script mirrors the data pipeline used by the Qwen LoRA trainers:
    • loads the HF DatasetDict created by `build_lspc_dataset.py`
    • tokenizes with the Qwen tokenizer (or any tokenizer you choose)
    • trains the compact EmbeddingClassifier using AdamW
    • reports macro/micro F1 + accuracy on validation and test splits

Example:
    python category_classification/train_embedding_classifier.py \\
        --data ./category_classification/lspc_dataset \\
        --out  ./embedding_classifier_prodcat
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from datasets import DatasetDict, load_from_disk
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.amp import autocast, GradScaler

# Ensure project root is importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
import sys

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from category_classification.embedding_classifier import (
    EmbeddingClassifier,
    EmbeddingClassifierConfig,
)

try:
    from ftlangdetect import detect as ft_detect

    def detect_language(text: str) -> str:
        text = text.strip()
        if not text:
            return "unk"
        try:
            res = ft_detect(text)
            return res["lang"] if isinstance(res, dict) else res.lang
        except Exception:
            return "unk"

    LANG_DETECTOR = "ftlangdetect"
except ModuleNotFoundError:
    try:
        from langdetect import detect as slow_detect  # type: ignore

        def detect_language(text: str) -> str:
            text = text.strip()
            if not text:
                return "unk"
            try:
                return slow_detect(text)
            except Exception:
                return "unk"

        LANG_DETECTOR = "langdetect"
    except ModuleNotFoundError:

        def detect_language(text: str) -> str:
            return "unk"

        LANG_DETECTOR = None


def bytes_to_gb(num_bytes: int) -> float:
    return num_bytes / (1024 ** 3)


def model_parameter_bytes(model: torch.nn.Module) -> int:
    return sum(p.nelement() * p.element_size() for p in model.parameters())


def optimizer_state_bytes(optimizer: torch.optim.Optimizer) -> int:
    total = 0
    for state in optimizer.state.values():
        for value in state.values():
            if torch.is_tensor(value):
                total += value.nelement() * value.element_size()
    return total


def gpu_memory_report(device: torch.device) -> str:
    if device.type != "cuda":
        return ""
    torch.cuda.synchronize(device)
    allocated = bytes_to_gb(torch.cuda.memory_allocated(device))
    reserved = bytes_to_gb(torch.cuda.memory_reserved(device))
    peak = bytes_to_gb(torch.cuda.max_memory_allocated(device))
    return f"alloc={allocated:.2f}GB reserved={reserved:.2f}GB peak={peak:.2f}GB"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EmbeddingClassifier on LSPC categories")
    p.add_argument("--data", type=str, default="./lspc_dataset",
                   help="Path to HF DatasetDict (train/validation/test)")
    p.add_argument("--out", type=str, default="./embedding_classifier_prodcat",
                   help="Directory to store checkpoints and tokenizer")
    p.add_argument("--seq_len", type=int, default=256, help="Max sequence length")
    p.add_argument("--batch_size", type=int, default=64, help="Per-device batch size")
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    p.add_argument("--dropout", type=float, default=0.1, help="Dropout probability")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs")
    p.add_argument("--resume", type=str, default=None,
                   help="Path to a checkpoint (checkpoint-epochX.pt) to resume from")
    p.add_argument("--no_lang_metrics", action="store_true",
                   help="Disable language-grouped metrics")
    p.add_argument("--lang_min_samples", type=int, default=1000,
                   help="Minimum samples per language to report metrics")
    p.add_argument("--lang_top_k", type=int, default=5,
                   help="Limit to top-K languages by frequency (0 disables)")
    p.add_argument("--amp", action="store_true",
                   help="Use torch.cuda.amp mixed precision on CUDA devices")
    p.add_argument("--memory_summary_freq", type=int, default=0,
                   help="Print torch.cuda.memory_summary() every N steps (0 disables)")
    return p.parse_args()


def resolve_path(path_str: str) -> Path:
    """Resolve dataset paths relative to CWD, script directory, or project root."""
    candidates = [
        Path(path_str),
        Path(__file__).resolve().parent / path_str,
        ROOT / path_str,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"Directory {path_str} not found (checked {candidates})")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def tokenize_dataset(
    ds: DatasetDict,
    tokenizer,
    seq_len: int,
) -> DatasetDict:
    def preprocess(batch: Dict[str, list]) -> Dict[str, list]:
        encoded = tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=seq_len,
        )
        encoded["labels"] = batch["label"]
        return encoded

    cols_to_remove = [c for c in ds["train"].column_names if c not in {"text", "label"}]
    ds_tok = ds.map(preprocess, batched=True, remove_columns=cols_to_remove)
    ds_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return ds_tok


def build_dataloaders(ds_tok: DatasetDict, batch_size: int) -> dict[str, DataLoader]:
    return {
        split: DataLoader(
            ds_tok[split],
            batch_size=batch_size,
            shuffle=(split == "train"),
            drop_last=False,
        )
        for split in ["train", "validation", "test"]
    }


def evaluate(
    model: EmbeddingClassifier,
    loader: DataLoader,
    device: torch.device,
    return_predictions: bool = False,
) -> dict[str, float] | Tuple[dict[str, float], np.ndarray, np.ndarray]:
    model.eval()
    preds, labels = [], []
    total_loss = 0.0
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out["loss"].detach()
            total_loss += loss.item() * batch["labels"].size(0)
            logits = out["logits"].detach().cpu().numpy()
            preds.append(np.argmax(logits, axis=-1))
            labels.append(batch["labels"].cpu().numpy())

    preds = np.concatenate(preds)
    labels = np.concatenate(labels)
    metrics = {
        "loss": total_loss / len(loader.dataset),
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "accuracy": accuracy_score(labels, preds),
    }
    if return_predictions:
        return metrics, preds, labels
    return metrics


def compute_language_metrics(
    labels: np.ndarray,
    preds: np.ndarray,
    texts: list[str],
    min_samples: int,
    top_k: int | None,
) -> dict[str, dict[str, float]]:
    indices_by_lang: dict[str, list[int]] = defaultdict(list)
    for idx, text in enumerate(texts):
        lang = detect_language(text)
        indices_by_lang[lang].append(idx)

    counts = Counter({lang: len(idxs) for lang, idxs in indices_by_lang.items()})
    sorted_langs = [lang for lang, cnt in counts.most_common() if cnt >= min_samples]
    if top_k:
        sorted_langs = sorted_langs[:top_k]

    lang_metrics: dict[str, dict[str, float]] = {}
    for lang in sorted_langs:
        idxs = np.array(indices_by_lang[lang], dtype=np.int64)
        lang_labels = labels[idxs]
        lang_preds = preds[idxs]
        lang_metrics[lang] = {
            "count": float(len(idxs)),
            "macro_f1": float(f1_score(lang_labels, lang_preds, average="macro", zero_division=0)),
            "micro_f1": float(f1_score(lang_labels, lang_preds, average="micro", zero_division=0)),
            "accuracy": float(accuracy_score(lang_labels, lang_preds)),
        }
    return lang_metrics


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"🖥️  Device: {device}")
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        print("⚡ AMP enabled (torch.cuda.amp)")

    data_path = resolve_path(args.data)
    print(f"📂 Loading dataset from: {data_path}")
    ds: DatasetDict = load_from_disk(str(data_path))
    num_labels = ds["train"].features["label"].num_classes

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    tokenizer.padding_side = "right"

    ds_tok = tokenize_dataset(ds, tokenizer, args.seq_len)
    loaders = build_dataloaders(ds_tok, args.batch_size)

    lang_metrics_enabled = (not args.no_lang_metrics) and LANG_DETECTOR
    if not LANG_DETECTOR:
        print("⚠️  Language detection libraries not installed; language metrics disabled.")
        lang_metrics_enabled = False
    elif args.no_lang_metrics:
        print("ℹ️  Language metrics explicitly disabled via CLI.")
    else:
        print(f"🗣️  Language metrics enabled via {LANG_DETECTOR}.")

    config = EmbeddingClassifierConfig(
        vocab_size=len(tokenizer),
        num_labels=num_labels,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = EmbeddingClassifier(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_param_bytes = model_parameter_bytes(model)
    print(f"🔢 Parameters: {total_params:,} total | {trainable_params:,} trainable")
    print(f"🧮 Model parameter memory ≈ {bytes_to_gb(model_param_bytes):.2f} GB")
    print("🧱 Model architecture:\n", model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = GradScaler(device="cuda") if use_amp else None
    opt_state_peak_bytes = optimizer_state_bytes(optimizer)
    opt_state_min_bytes = opt_state_peak_bytes or float("inf")
    if opt_state_peak_bytes:
        print(f"🧮 Optimizer state memory ≈ {bytes_to_gb(opt_state_peak_bytes):.2f} GB (initial)")

    best_macro_f1 = -1.0
    best_state = None
    global_step = 0
    start_epoch = 1

    if args.resume:
        ckpt_path = resolve_path(args.resume)
        print(f"♻️  Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        best_macro_f1 = checkpoint.get("best_macro_f1", -1.0)
        global_step = checkpoint.get("global_step", 0)
        start_epoch = checkpoint["epoch"] + 1
        torch_rng = checkpoint["torch_rng_state"]
        if isinstance(torch_rng, torch.Tensor):
            torch_rng = torch_rng.detach().clone().to(dtype=torch.uint8, device="cpu").contiguous()
        else:
            torch_rng = torch.tensor(torch_rng, dtype=torch.uint8)
        torch.set_rng_state(torch_rng)
        if torch.cuda.is_available() and "cuda_rng_state" in checkpoint:
            cuda_rng = checkpoint["cuda_rng_state"]
            if isinstance(cuda_rng, torch.Tensor):
                cuda_rng = cuda_rng.detach().clone().to(dtype=torch.uint8, device="cpu").contiguous()
            else:
                cuda_rng = torch.tensor(cuda_rng, dtype=torch.uint8)
            torch.cuda.set_rng_state(cuda_rng)
        np.random.set_state(checkpoint["numpy_rng_state"])
        print(f"➡️  Resumed from epoch {checkpoint['epoch']}, best macro-f1 {best_macro_f1:.4f}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    gpu_min_alloc_bytes = float("inf") if device.type == "cuda" else 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)

        last_step = 0
        for step, batch in enumerate(loaders["train"], start=1):
            last_step = step
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast(device_type="cuda", enabled=use_amp):
                out = model(**batch)
                loss = out["loss"] / args.grad_accum
            if use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total_loss += loss.item()

            if step % args.grad_accum == 0:
                if use_amp:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                if use_amp:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step % 5000 == 0:
                avg_loss = total_loss / step
                opt_state = optimizer_state_bytes(optimizer)
                opt_state_peak_bytes = max(opt_state_peak_bytes, opt_state)
                opt_state_min_bytes = min(opt_state_min_bytes, opt_state)
                mem_info = gpu_memory_report(device)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                    curr_alloc = torch.cuda.memory_allocated(device)
                    gpu_min_alloc_bytes = min(gpu_min_alloc_bytes, curr_alloc)
                msg = (
                    f"Epoch {epoch} | Step {step} | Loss {avg_loss:.4f} | "
                    f"weights={bytes_to_gb(model_param_bytes):.2f}GB "
                    f"opt_state={bytes_to_gb(opt_state):.2f}GB "
                    f"(min {bytes_to_gb(opt_state_min_bytes):.2f} / "
                    f"max {bytes_to_gb(opt_state_peak_bytes):.2f})"
                )
                if mem_info:
                    msg += (
                        f" | GPU {mem_info} "
                        f"(min_alloc={bytes_to_gb(gpu_min_alloc_bytes):.2f}GB)"
                    )
                print(msg)
                if args.memory_summary_freq and step % args.memory_summary_freq == 0 and device.type == "cuda":
                    print(torch.cuda.memory_summary(device=device))

        if last_step and last_step % args.grad_accum != 0:
            if use_amp:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if use_amp:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        val_eval = evaluate(
            model,
            loaders["validation"],
            device,
            return_predictions=lang_metrics_enabled,
        )
        if lang_metrics_enabled:
            val_metrics, val_preds, val_labels = val_eval
        else:
            val_metrics = val_eval
        print(f"Epoch {epoch} | Validation: {val_metrics}")
        if val_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = val_metrics["macro_f1"]
            best_state = {
                "model_state": model.state_dict(),
                "config": config.__dict__,
                "tokenizer_dir": None,
            }
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if args.save_every > 0 and epoch % args.save_every == 0:
            ckpt = {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_macro_f1": best_macro_f1,
                "global_step": global_step,
                "torch_rng_state": torch.get_rng_state(),
                "numpy_rng_state": np.random.get_state(),
            }
            if torch.cuda.is_available():
                ckpt["cuda_rng_state"] = torch.cuda.get_rng_state()
            ckpt_path = out_dir / f"checkpoint-epoch{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        if lang_metrics_enabled:
            lang_texts = ds["validation"]["text"]
            top_k = args.lang_top_k if args.lang_top_k > 0 else None
            val_language_metrics = compute_language_metrics(
                labels=val_labels,
                preds=val_preds,
                texts=lang_texts,
                min_samples=args.lang_min_samples,
                top_k=top_k,
            )
            if val_language_metrics:
                lang_path = out_dir / f"lang_metrics_validation_epoch{epoch}.json"
                with open(lang_path, "w", encoding="utf-8") as fh:
                    json.dump(
                        {"epoch": epoch, "language_metrics": val_language_metrics},
                        fh,
                        indent=2,
                    )
                print(f"📝 Saved validation language metrics → {lang_path.name}")

    if best_state:
        model.load_state_dict(best_state["model_state"])

    test_eval = evaluate(
        model, loaders["test"], device, return_predictions=lang_metrics_enabled
    )
    if lang_metrics_enabled:
        test_metrics, test_preds, test_labels = test_eval
    else:
        test_metrics = test_eval
        test_preds = test_labels = None
    print(f"Test metrics: {test_metrics}")

    language_metrics = None
    if lang_metrics_enabled and test_labels is not None and test_preds is not None:
        print("🗣️  Computing language-grouped metrics...")
        lang_texts = ds["test"]["text"]
        top_k = args.lang_top_k if args.lang_top_k > 0 else None
        language_metrics = compute_language_metrics(
            labels=test_labels,
            preds=test_preds,
            texts=lang_texts,
            min_samples=args.lang_min_samples,
            top_k=top_k,
        )
        if not language_metrics:
            print("ℹ️  No languages met the minimum sample threshold.")
        else:
            for lang, metrics in language_metrics.items():
                print(
                    f"   [{lang}] count={int(metrics['count'])} "
                    f"macro_f1={metrics['macro_f1']:.3f} "
                    f"micro_f1={metrics['micro_f1']:.3f} "
                    f"acc={metrics['accuracy']:.3f}"
                )

    torch.save(model.state_dict(), out_dir / "embedding_classifier.pt")
    with open(out_dir / "config.json", "w", encoding="utf-8") as fh:
        json.dump(config.__dict__, fh, indent=2)
    tokenizer.save_pretrained(out_dir / "tokenizer")
    metrics_payload = {"validation_macro_f1": best_macro_f1, "test": test_metrics}
    if language_metrics:
        metrics_payload["language_metrics"] = language_metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)
    print(f"✅ Training complete. Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
