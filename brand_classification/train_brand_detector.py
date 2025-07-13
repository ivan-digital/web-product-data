#!/usr/bin/env python3
"""
Fine-tune Qwen/Qwen2-0.5B with LoRA for brand detection.

This script trains a binary classifier to determine if a brand is present in the product text.
"""
from __future__ import annotations
import argparse, os, torch
from datasets import load_from_disk, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score, precision_recall_fscore_support
import numpy as np

# ────────────────────────────────────────────────────────────────────────────────
def infer_device_dtype() -> tuple[torch.device, torch.dtype]:
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32

# ────────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--model_name_or_path", type=str, default="qwen/qwen2-0.5b")
ap.add_argument("--dataset_name", type=str, default="./brand_dataset")
ap.add_argument("--output_dir", type=str, default="./qwen3_lora_brand_detector")
ap.add_argument("--num_train_epochs", type=int, default=3)
ap.add_argument("--per_device_train_batch_size", type=int, default=16)
ap.add_argument("--gradient_accumulation_steps", type=int, default=2)
ap.add_argument("--learning_rate", type=float, default=5e-5)
ap.add_argument("--lora_r", type=int, default=32)
ap.add_argument("--lora_alpha", type=int, default=64)
ap.add_argument("--seq_len", type=int, default=256)
ap.add_argument("--max_grad_norm", type=float, default=0.5)
ap.add_argument("--weight_decay", type=float, default=0.01)
ap.add_argument("--lr_scheduler_type", type=str, default="cosine")
ap.add_argument("--warmup_ratio", type=float, default=0.05)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--fp16", action="store_true")
args = ap.parse_args()

# ────────────────────────────────────────────────────────────────────────────────
device, dtype = infer_device_dtype()
is_cuda = device.type == "cuda"
is_mps  = device.type == "mps"

print(f"🖥️  Running on: {device} (training dtype: {dtype})")

if is_mps:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")

# ────────────────────────────────────────────────────────────────────────────────
ds: DatasetDict = load_from_disk(args.dataset_name)

labels     = ds["train"].features["label"].names
label2id   = {l: i for i, l in enumerate(labels)}
id2label   = {i: l for l, i in label2id.items()}

tok = AutoTokenizer.from_pretrained(
    args.model_name_or_path,
    trust_remote_code=True,
)
if tok.pad_token_id is None:
    tok.pad_token = tok.eos_token or "<|pad|>"
tok.padding_side = "right"

def preprocess(batch: dict) -> dict:
    tokenized = tok(
        batch["text"],
        max_length=args.seq_len,
        padding="max_length",
        truncation=True,
    )
    tokenized["label"] = batch["label"]
    return tokenized

ds_tok = ds.map(preprocess, batched=True,
                remove_columns=ds["train"].column_names)

# ────────────────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name_or_path,
    num_labels=len(labels),
    id2label=id2label,
    label2id=label2id,
    trust_remote_code=True,
)
model.to(device)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id
if len(tok) != model.config.vocab_size:
    model.resize_token_embeddings(len(tok))

MODULES_TO_SAVE = ["score"]
target_modules  = ["q_proj","k_proj","v_proj","o_proj",
                       "gate_proj","up_proj","down_proj"]
lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules,
    modules_to_save=MODULES_TO_SAVE,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    return {
        'accuracy': accuracy_score(labels, preds),
        'f1': f1,
        'precision': precision,
        'recall': recall
    }

training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=args.fp16,
    optim="adamw_torch",
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    logging_steps=100,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    gradient_checkpointing=True,
    report_to="none",
    seed=args.seed,
)

data_collator = DataCollatorWithPadding(tokenizer=tok)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
)

trainer.train()

print("💾 Saving model...")
trainer.save_model(args.output_dir)

# Evaluate on the test set
print("🧪 Evaluating on test set...")
test_results = trainer.predict(ds_tok["test"])
print(test_results.metrics)
