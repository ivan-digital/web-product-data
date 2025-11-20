#!/usr/bin/env python3
"""
Training entrypoint for EmbeddingClassifierV2.

Adds classifier-head regularisation options plus label smoothing / focal
loss controls on top of the baseline trainer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict
import sys

import numpy as np
import torch
import torch.nn.functional as F
from datasets import DatasetDict, load_from_disk
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# Ensure project root importable when running as a script
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from category_classification.embedding_classifier_v2 import (
    EmbeddingClassifierV2,
    EmbeddingClassifierV2Config,
)
from category_classification.train_embedding_classifier import (
    LANG_DETECTOR,
    build_dataloaders,
    build_scheduler,
    bytes_to_gb,
    compute_language_metrics,
    evaluate,
    get_device,
    gpu_memory_report,
    model_parameter_bytes,
    optimizer_state_bytes,
    resolve_path,
    set_seed,
    tokenize_dataset,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train EmbeddingClassifierV2 with enhanced regularisation.")
    p.add_argument("--data", type=str, default="./lspc_dataset_full", help="Path to HF DatasetDict.")
    p.add_argument("--out", type=str, default="./embedding_classifier_prodcat_v2", help="Output directory.")
    p.add_argument("--seq_len", type=int, default=256, help="Max token length.")
    p.add_argument("--batch_size", type=int, default=64, help="Per-device batch size.")
    p.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps.")
    p.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    p.add_argument("--lr", type=float, default=2e-4, help="Learning rate.")
    p.add_argument("--lr_schedule", choices=["none", "cosine", "exponential"], default="cosine", help="Scheduler type.")
    p.add_argument("--lr_warmup_steps", type=int, default=2000, help="Warmup steps.")
    p.add_argument("--lr_min_scale", type=float, default=0.05, help="Final LR fraction for schedulers.")
    p.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay.")
    p.add_argument("--dropout", type=float, default=0.1, help="Backbone dropout.")
    p.add_argument("--classifier_hidden", type=int, default=768, help="Classifier bottleneck size.")
    p.add_argument("--classifier_dropout", type=float, default=0.15, help="Classifier dropout.")
    p.add_argument("--classifier_layernorm_eps", type=float, default=1e-6, help="Classifier RMSNorm epsilon.")
    p.add_argument("--classifier_no_residual", action="store_true", help="Disable residual connection in head.")
    p.add_argument("--pooler", choices=["mean", "cls-first-token"], default="mean", help="Pooling strategy.")
    p.add_argument("--label_smoothing", type=float, default=0.05, help="Label smoothing factor.")
    p.add_argument("--focal_gamma", type=float, default=0.0, help="Focal loss gamma (0 disables).")
    p.add_argument("--focal_alpha", type=float, default=1.0, help="Focal loss alpha multiplier.")
    p.add_argument("--seed", type=int, default=42, help="Random seed.")
    p.add_argument("--save_every", type=int, default=1, help="Save checkpoint every N epochs.")
    p.add_argument("--resume", type=str, default=None, help="Checkpoint to resume from.")
    p.add_argument("--no_lang_metrics", action="store_true", help="Disable language metrics.")
    p.add_argument("--lang_min_samples", type=int, default=1000, help="Minimum samples per language.")
    p.add_argument("--lang_top_k", type=int, default=5, help="Top-K languages to log (0 disables).")
    p.add_argument("--amp", action="store_true", help="Use AMP on CUDA.")
    p.add_argument("--memory_summary_freq", type=int, default=0, help="cuda.memory_summary frequency.")
    return p.parse_args()


def classification_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    label_smoothing: float,
    focal_gamma: float,
    focal_alpha: float,
) -> torch.Tensor:
    losses = F.cross_entropy(
        logits,
        labels,
        reduction="none",
        label_smoothing=max(0.0, label_smoothing),
    )
    if focal_gamma > 0.0:
        probs = torch.softmax(logits, dim=-1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1).clamp(min=1e-6, max=1.0)
        focal_weight = (1.0 - pt) ** focal_gamma
        if focal_alpha != 1.0:
            focal_weight = focal_weight * focal_alpha
        losses = losses * focal_weight
    return losses.mean()


def save_language_metrics(
    path: Path,
    epoch: int,
    labels: np.ndarray,
    preds: np.ndarray,
    texts: list[str],
    min_samples: int,
    top_k: int | None,
    languages: list[str] | None = None,
) -> None:
    language_metrics = compute_language_metrics(
        labels=labels,
        preds=preds,
        texts=texts,
        min_samples=min_samples,
        top_k=top_k,
        languages=languages,
    )
    if not language_metrics:
        return
    payload = {"epoch": epoch, "language_metrics": language_metrics}
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    print(f"Saved Saved validation language metrics -> {path.name}")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    use_amp = args.amp and device.type == "cuda"
    if use_amp:
        print("AMP enabled (torch.cuda.amp)")

    data_path = resolve_path(args.data)
    print(f"Loading dataset from Loading dataset from: {data_path}")
    ds: DatasetDict = load_from_disk(str(data_path))
    num_labels = ds["train"].features["label"].num_classes

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Embedding-0.6B", trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token or "<|pad|>"
    tokenizer.padding_side = "right"

    ds_tok = tokenize_dataset(ds, tokenizer, args.seq_len)
    loaders = build_dataloaders(ds_tok, args.batch_size)
    steps_per_epoch = max(1, (len(loaders["train"]) + args.grad_accum - 1) // args.grad_accum)
    total_updates = steps_per_epoch * args.epochs

    lang_metrics_enabled = (not args.no_lang_metrics) and LANG_DETECTOR
    if not LANG_DETECTOR:
        print("Warning:  Language detection libraries not installed; language metrics disabled.")
        lang_metrics_enabled = False
    elif args.no_lang_metrics:
        print("Info:  Language metrics explicitly disabled via CLI.")
    else:
        print(f"Language metrics  Language metrics enabled via {LANG_DETECTOR}.")

    config = EmbeddingClassifierV2Config(
        vocab_size=len(tokenizer),
        num_labels=num_labels,
        dropout=args.dropout,
        pad_token_id=tokenizer.pad_token_id,
        classifier_hidden_size=args.classifier_hidden,
        classifier_dropout=args.classifier_dropout,
        classifier_layernorm_eps=args.classifier_layernorm_eps,
        classifier_residual=not args.classifier_no_residual,
        pooler=args.pooler,
    )
    model = EmbeddingClassifierV2(config).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model_param_bytes = model_parameter_bytes(model)
    print(f"Parameters: Parameters: {total_params:,} total | {trainable_params:,} trainable")
    print(f"Model parameter memory ~ Model parameter memory ~ {bytes_to_gb(model_param_bytes):.2f} GB")
    print("Model architecture: Model architecture:\n", model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = build_scheduler(
        optimizer=optimizer,
        schedule_type=args.lr_schedule,
        total_updates=total_updates,
        warmup_steps=args.lr_warmup_steps,
        min_scale=args.lr_min_scale,
    )
    if scheduler:
        print(
            "Scheduler  LR scheduler: "
            f"{args.lr_schedule} | warmup={args.lr_warmup_steps} | "
            f"min_scale={args.lr_min_scale} | total_updates={total_updates}"
        )
    else:
        print("Scheduler  LR scheduler: none (constant LR)")
    scaler = GradScaler(device="cuda") if use_amp else None

    opt_state_peak_bytes = optimizer_state_bytes(optimizer)
    opt_state_min_bytes = opt_state_peak_bytes or float("inf")
    best_macro_f1 = -1.0
    best_state = None
    global_step = 0
    start_epoch = 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    scheduler_needs_backfill = False
    if args.resume:
        ckpt_path = resolve_path(args.resume)
        print(f"Resuming from checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        sched_state = checkpoint.get("scheduler_state")
        if scheduler and sched_state:
            scheduler.load_state_dict(sched_state)
            last_lrs = scheduler.get_last_lr()
            for group, lr in zip(optimizer.param_groups, last_lrs):
                group["lr"] = lr
        elif scheduler:
            scheduler_needs_backfill = True
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
        if scheduler and scheduler_needs_backfill and global_step > 0:
            scheduler.step(global_step - 1)
        print(f"Resumed from epoch {checkpoint['epoch']}, best macro-f1 {best_macro_f1:.4f}")

    gpu_min_alloc_bytes = float("inf") if device.type == "cuda" else 0.0

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        total_loss = 0.0
        last_step = 0
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loaders["train"], start=1):
            last_step = step
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch["labels"]
            with autocast(device_type="cuda", enabled=use_amp):
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                )
                loss = classification_loss(
                    outputs["logits"],
                    labels,
                    args.label_smoothing,
                    args.focal_gamma,
                    args.focal_alpha,
                ) / args.grad_accum
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
                if scheduler:
                    scheduler.step()

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
                curr_lr = optimizer.param_groups[0]["lr"]
                msg = (
                    f"Epoch {epoch} | Step {step} | Loss {avg_loss:.4f} | "
                    f"weights={bytes_to_gb(model_param_bytes):.2f}GB "
                    f"opt_state={bytes_to_gb(opt_state):.2f}GB "
                    f"(min {bytes_to_gb(opt_state_min_bytes):.2f} / "
                    f"max {bytes_to_gb(opt_state_peak_bytes):.2f}) | lr={curr_lr:.6f}"
                )
                if mem_info:
                    msg += f" | GPU {mem_info} (min_alloc={bytes_to_gb(gpu_min_alloc_bytes):.2f}GB)"
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
            global_step += 1
            if scheduler:
                scheduler.step()

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
                "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                "config": config.__dict__,
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
            if scheduler:
                ckpt["scheduler_state"] = scheduler.state_dict()
            ckpt_path = out_dir / f"checkpoint-epoch{epoch}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"Saved Saved checkpoint: {ckpt_path}")

        if lang_metrics_enabled:
            lang_texts = ds["validation"]["text"]
            lang_codes = (
                ds["validation"]["language"]
                if "language" in ds["validation"].column_names
                else None
            )
            top_k = args.lang_top_k if args.lang_top_k > 0 else None
            save_language_metrics(
                out_dir / f"lang_metrics_validation_epoch{epoch}.json",
                epoch,
                labels=val_labels,
                preds=val_preds,
                texts=lang_texts,
                min_samples=args.lang_min_samples,
                top_k=top_k,
                languages=lang_codes,
            )

    if best_state:
        print(f"Reloading Loading best validation macro-F1 weights ({best_macro_f1:.4f}).")
        model.load_state_dict(best_state["model_state"])

    test_eval = evaluate(
        model,
        loaders["test"],
        device,
        return_predictions=lang_metrics_enabled,
    )
    if lang_metrics_enabled:
        test_metrics, test_preds, test_labels = test_eval
    else:
        test_metrics = test_eval
        test_preds = test_labels = None
    print(f"Test metrics: {test_metrics}")

    language_metrics = None
    if lang_metrics_enabled and test_labels is not None and test_preds is not None:
        print("Language metrics  Computing language-grouped metrics...")
        lang_texts = ds["test"]["text"]
        lang_codes = (
            ds["test"]["language"] if "language" in ds["test"].column_names else None
        )
        top_k = args.lang_top_k if args.lang_top_k > 0 else None
        language_metrics = compute_language_metrics(
            labels=test_labels,
            preds=test_preds,
            texts=lang_texts,
            min_samples=args.lang_min_samples,
            top_k=top_k,
            languages=lang_codes,
        )
        if not language_metrics:
            print("Info:  No languages met the minimum sample threshold.")
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
    metrics_payload: Dict[str, object] = {"best_validation_macro_f1": best_macro_f1, "test": test_metrics}
    if language_metrics:
        metrics_payload["language_metrics"] = language_metrics
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics_payload, fh, indent=2)
    print(f"Training complete Training complete. Artifacts saved to {out_dir}")


if __name__ == "__main__":
    main()
