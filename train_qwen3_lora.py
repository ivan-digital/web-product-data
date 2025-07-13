#!/usr/bin/env python3
"""
Fine‑tune Qwen/Qwen3‑Embedding‑0.6B with LoRA for 24‑class product
categorisation – macro‑F1 eval, streaming‑friendly, optional 4‑bit
quantisation.

Runs on:
    • CUDA  : FP16 (+ optional 4‑bit)
    • Apple : FP32   – automatic micro‑batch to avoid OOM
    • CPU   : FP32

Example
-------
python train_qwen3_lora.py \
    --data ./lspc_dataset \
    --out  ./qwen3_lora_prodcat \
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
# Helpers: device, dtype, micro‑batch
# ────────────────────────────────────────────────────────────────────────────────
def infer_device_dtype() -> tuple[torch.device, torch.dtype]:
    """
    Decide device and *training* compute dtype.

    * CUDA   → FP16 (speed + RAM) – fully supported
    * MPS/CPU → FP32 – half precision kernels missing / slow
    """
    if torch.cuda.is_available():
        return torch.device("cuda"), torch.float16
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.float32
    return torch.device("cpu"), torch.float32


def adjust_micro_batch(effective_bs: int, device: torch.device) -> tuple[int, int]:
    """Return (real_micro_batch, grad_accum_steps) safe for the given device."""
    if device.type == "cuda":
        return effective_bs, 1

    limit = 8 if device.type == "mps" else 4
    if effective_bs <= limit:
        return effective_bs, 1

    grad_accum = max(1, effective_bs // limit)
    print(
        f"⚠️  Batch {effective_bs} too large for {device}; "
        f"using micro‑batch {limit} with {grad_accum} gradient‑accumulation steps."
    )
    return limit, grad_accum


# ────────────────────────────────────────────────────────────────────────────────
# CLI
# ────────────────────────────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument("--data",  default="./lspc_dataset",
                help="HF DatasetDict with train/validation/test")
ap.add_argument("--out",   default="./qwen3_lora_prodcat",
                help="Output checkpoint directory")
ap.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B",
                help="Base model repo or path")
ap.add_argument("--batch",  type=int, default=32,
                help="Effective (logical) batch size")
ap.add_argument("--epochs", type=int, default=1)
ap.add_argument("--lr",     type=float, default=5e-5,
                help="LoRA learning rate")
ap.add_argument("--lora_r",     type=int, default=16)
ap.add_argument("--lora_alpha", type=int, default=32)
ap.add_argument("--seq_len",    type=int, default=256,
                help="Max sequence length")
ap.add_argument("--balance_to_min_class", action="store_true",
                help="Downsample training set to the size of the smallest class")
ap.add_argument("--max_samples_per_class", type=int, default=None,
                help="Downsample training set to this many samples per class. "
                     "Overrides --balance_to_min_class.")
args = ap.parse_args()


# ────────────────────────────────────────────────────────────────────────────────
# Environment, device, dtype
# ────────────────────────────────────────────────────────────────────────────────
device, dtype = infer_device_dtype()
is_cuda = device.type == "cuda"
is_mps  = device.type == "mps"

print(f"🖥️  Running on: {device} (training dtype: {dtype})")

# Allow PyTorch to fall back on CPU ops that the MPS backend lacks
if is_mps:
    os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Prevent Apple’s 80 % memory ceiling unless user already set it
if is_mps:
    os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "0.0")


# ────────────────────────────────────────────────────────────────────────────────
# Gradient accumulation / micro‑batch
# ────────────────────────────────────────────────────────────────────────────────
micro_batch, grad_accum = adjust_micro_batch(args.batch, device)


# ────────────────────────────────────────────────────────────────────────────────
# Dataset
# ────────────────────────────────────────────────────────────────────────────────
ds: DatasetDict = load_from_disk(args.data)          # expects train/valid/test

# Balanced (down)sampling of training set
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

    # Group indices by class & shuffle them
    indices_by_label = {}
    torch.manual_seed(42)
    for i, label in enumerate(train_ds["label"]):
        if label not in indices_by_label:
            indices_by_label[label] = []
        indices_by_label[label].append(i)
    for label in indices_by_label:
        perm = torch.randperm(len(indices_by_label[label]))
        indices_by_label[label] = [indices_by_label[label][i] for i in perm]

    # Sample indices from each class
    balanced_indices = []
    for label, indices in indices_by_label.items():
        sampled_indices = indices[:sample_size]
        balanced_indices.extend(sampled_indices)

    # Create the new balanced dataset from the selected indices
    ds["train"] = train_ds.select(balanced_indices)
    print(f"   New training set size: {len(ds['train'])}")


labels     = sorted({l for l in ds["train"]["label"]})
label2id   = {l: i for i, l in enumerate(labels)}
id2label   = {i: l for l, i in label2id.items()}

tok = AutoTokenizer.from_pretrained(
    args.model,
    token="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",
    trust_remote_code=True,
)
if tok.pad_token_id is None:                         # Qwen3 has no pad by default
    tok.pad_token = tok.eos_token or "<|pad|>"
tok.padding_side = "right"

# Pre‑process: tokenize text, map labels to IDs
# Note: we don't need to one‑hot encode, HF handles that
def preprocess(batch: dict) -> dict:
    # The tokenizer will return input_ids, attention_mask, etc.
    tokenized = tok(
        batch["text"],
        max_length=args.seq_len,
        padding="max_length",
        truncation=True,
    )
    tokenized["label"] = batch["label"]
    return tokenized

# The `map` function is lazy, so this is fast
ds_tok = ds.map(preprocess, batched=True,
                remove_columns=ds["train"].column_names)


# ────────────────────────────────────────────────────────────────────────────────
# Optional 4‑bit quant (CUDA only)
# ────────────────────────────────────────────────────────────────────────────────
bnb_cfg = None
if is_cuda and importlib.util.find_spec("bitsandbytes"):
    from transformers import BitsAndBytesConfig
    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )


# ────────────────────────────────────────────────────────────────────────────────
# Model
# ────────────────────────────────────────────────────────────────────────────────
model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    token="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",
    num_labels=len(labels),
    device_map="auto" if is_cuda else None,
    torch_dtype=dtype,
    quantization_config=bnb_cfg,
    trust_remote_code=True,
)

# Ensure tokenizer & model agree on special tokens
if model.config.pad_token_id is None:
    model.config.pad_token_id = tok.pad_token_id
if len(tok) != model.config.vocab_size:
    model.resize_token_embeddings(len(tok))

# ────────────────────────────────────────────────────────────────────────────────
# LoRA
# ────────────────────────────────────────────────────────────────────────────────
def find_linear_names(m, blacklist: set[str] | None = None) -> list[str]:
    blacklist = blacklist or set()
    names = set()
    for full_name, mod in m.named_modules():
        if isinstance(mod, torch.nn.Linear):
            leaf = full_name.split(".")[-1]
            if leaf not in blacklist:
                names.add(leaf)
    return sorted(names)

MODULES_TO_SAVE = ["score"]                         # keep CLS head outside LoRA
target_modules  = find_linear_names(model, set(MODULES_TO_SAVE))

lora_cfg = LoraConfig(
    r=args.lora_r,
    lora_alpha=args.lora_alpha,
    target_modules=target_modules,
    modules_to_save=MODULES_TO_SAVE,
    lora_dropout=0.2,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, lora_cfg)
# Print stats
model.print_trainable_parameters()


# Metrics: macro F1
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "accuracy": accuracy_score(labels, preds),
    }


# ────────────────────────────────────────────────────────────────────────────────
# TrainingArguments
# ────────────────────────────────────────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir=args.out,
    # Training strategy
    num_train_epochs=args.epochs,
    per_device_train_batch_size=micro_batch,
    per_device_eval_batch_size=micro_batch,
    gradient_accumulation_steps=grad_accum,
    # Dtype
    fp16=False,
    bf16=False,
    # Optimisation
    optim="paged_adamw_8bit" if is_cuda else "adamw_torch",
    learning_rate=args.lr,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    weight_decay=0.01,
    max_grad_norm=1.0,
    logging_steps=50,
    save_strategy="epoch",
    eval_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="macro_f1",
    gradient_checkpointing=True,
    report_to="none",
    seed=42,
)


# ────────────────────────────────────────────────────────────────────────────────
# Run training
# ────────────────────────────────────────────────────────────────────────────────
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

# Save final model
val_metrics = trainer.evaluate(ds_tok["validation"])
print("Validation:", val_metrics)
test_metrics = trainer.evaluate(ds_tok["test"])
print("Test:", test_metrics)


# ────────────────────────────────────────────────────────────────────────────────
# Confusion Matrix
# ────────────────────────────────────────────────────────────────────────────────
print("\nCalculating Confusion Matrix for Test Set...")
predictions = trainer.predict(ds_tok["test"])
y_pred = np.argmax(predictions.predictions, axis=1)
y_true = predictions.label_ids
cm = confusion_matrix(y_true, y_pred)

# Get class names for plotting
class_names = [id2label[i] for i in range(len(id2label))]

print("\nConfusion Matrix:")
# Pretty print using pandas if available
try:
    import pandas as pd
    cm_df = pd.DataFrame(cm, index=class_names, columns=class_names)
    print(cm_df)
except ImportError:
    print("(Install pandas for a prettier confusion matrix)")
    print(cm)


# ────────────────────────────────────────────────────────────────────────────────
# Merge LoRA + base and save a single inference‑ready checkpoint
# ────────────────────────────────────────────────────────────────────────────────
trainer.model.merge_and_unload()
merged_dir = os.path.join(args.out, "merged")
trainer.model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
print(f"✅ finished – merged model at {merged_dir}")