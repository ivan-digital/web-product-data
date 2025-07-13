#!/usr/bin/env python3
"""
Fine‑tune Qwen/Qwen3‑Embedding‑0.6B with LoRA for 24‑class product
categorisation – macro‑F1 eval, streaming‑friendly, optional 4‑bit
quantisation.

This version uses the classic Adam optimizer.

Runs on:
    • CUDA  : FP16 (+ optional 4‑bit)
    • Apple : FP32   – automatic micro‑batch to avoid OOM
    • CPU   : FP32

Example
-------
python train_qwen3_lora_adam.py \
    --data ./lspc_dataset \
    --out  ./qwen3_lora_prodcat_adam \
    --batch 128
"""
from __future__ import annotations
import argparse, os, inspect, importlib.util, warnings, torch
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
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix
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
ap.add_argument("--dataset_name", type=str, default="./lspc_dataset")
ap.add_argument("--output_dir", type=str, default="./qwen3_lora_prodcat_adam")
ap.add_argument("--num_train_epochs", type=int, default=1)
ap.add_argument("--per_device_train_batch_size", type=int, default=4)
ap.add_argument("--gradient_accumulation_steps", type=int, default=8)
ap.add_argument("--learning_rate", type=float, default=5e-5)
ap.add_argument("--lora_r", type=int, default=32)
ap.add_argument("--lora_alpha", type=int, default=64)
ap.add_argument("--seq_len", type=int, default=256)
ap.add_argument("--balance_to_min_class", action="store_true")
ap.add_argument("--max_samples_per_class", type=int, default=None)
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
if is_mps:
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


# ────────────────────────────────────────────────────────────────────────────────
ds: DatasetDict = load_from_disk(args.dataset_name)
sample_size = args.max_samples_per_class
if not sample_size and args.balance_to_min_class:
    from collections import Counter
    print("⚖️  Balancing training set to smallest class size...")
    counts = Counter(ds["train"]["label"])
    sample_size = min(counts.values())
    print(f"   Smallest class has {sample_size} samples.")
if sample_size:
    print(f"   Downsampling to {sample_size} samples per class...")
    train_ds = ds["train"]
    indices_by_label = {}
    torch.manual_seed(42)
    for i, label in enumerate(train_ds["label"]):
        if label not in indices_by_label:
            indices_by_label[label] = []
        indices_by_label[label].append(i)
    for label in indices_by_label:
        perm = torch.randperm(len(indices_by_label[label]))
        indices_by_label[label] = [indices_by_label[label][i] for i in perm]
    balanced_indices = []
    for label, indices in indices_by_label.items():
        sampled_indices = indices[:sample_size]
        balanced_indices.extend(sampled_indices)
    ds["train"] = train_ds.select(balanced_indices)
    print(f"   New training set size: {len(ds['train'])}")
labels     = sorted({l for l in ds["train"]["label"]})
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
    trust_remote_code=True,
)
model.to(device)

if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id
if len(tok) != model.config.vocab_size:
    model.resize_token_embeddings(len(tok))

def find_linear_names(m, blacklist: set[str] | None = None) -> list[str]:
    blacklist = blacklist or set()
    names = set()
    for full_name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear):
            leaf = full_name.split(".")[-1]
            if leaf not in blacklist:
                names.add(leaf)
    return sorted(names)
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
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "accuracy": accuracy_score(labels, preds),
    }
training_args = TrainingArguments(
    output_dir=args.output_dir,
    num_train_epochs=args.num_train_epochs,
    per_device_train_batch_size=args.per_device_train_batch_size,
    per_device_eval_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    fp16=args.fp16,
    bf16=False,
    optim="adamw_torch",
    learning_rate=args.learning_rate,
    lr_scheduler_type=args.lr_scheduler_type,
    warmup_ratio=args.warmup_ratio,
    weight_decay=args.weight_decay,
    max_grad_norm=args.max_grad_norm,
    logging_steps=1000,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    gradient_checkpointing=True,
    report_to="none",
    seed=args.seed,
    fp16_full_eval=args.fp16,
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
val_metrics = trainer.evaluate(ds_tok["validation"])
print("Validation:", val_metrics)
test_metrics = trainer.evaluate(ds_tok["test"])
print("Test:", test_metrics)
print("\nCalculating Confusion Matrix for Test Set...")
predictions = trainer.predict(ds_tok["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
cm = confusion_matrix(y_true, y_pred)
class_names = [id2label[i] for i in range(len(id2label))]
print("\nConfusion Matrix:")
try:
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
except ImportError:
    print("(Install pandas for a prettier confusion matrix)")
    print(cm)
trainer.model.merge_and_unload()
merged_dir = os.path.join(args.output_dir, "merged")
trainer.model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
print(f"✅ finished – merged model at {merged_dir}")
