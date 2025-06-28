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
)
from peft import LoraConfig, get_peft_model, TaskType
from sklearn.metrics import f1_score, accuracy_score


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
ap.add_argument("--batch",  type=int, default=128,
                help="Effective (logical) batch size")
ap.add_argument("--epochs", type=int, default=2)
ap.add_argument("--lr",     type=float, default=2e-4,
                help="LoRA learning rate")
ap.add_argument("--lora_r",     type=int, default=16)
ap.add_argument("--lora_alpha", type=int, default=32)
ap.add_argument("--seq_len",    type=int, default=256,
                help="Max sequence length")
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
labels     = sorted({l for l in ds["train"]["label"]})
label2id   = {l: i for i, l in enumerate(labels)}
id2label   = {i: l for l, i in label2id.items()}

tok = AutoTokenizer.from_pretrained(
    args.model,
    token="hf_your_token_if_needed",
    trust_remote_code=True,
)
if tok.pad_token_id is None:                         # Qwen3 has no pad by default
    tok.pad_token = tok.eos_token or "<|pad|>"
tok.padding_side = "right"

def preprocess(batch):
    enc = tok(batch["text"],
              truncation=True,
              padding="max_length",
              max_length=args.seq_len)
    enc["labels"] = [label2id[l] for l in batch["label"]]
    return enc

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
    token="hf_your_token_if_needed",
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
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.SEQ_CLS,
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


# ────────────────────────────────────────────────────────────────────────────────
# Metrics
# ────────────────────────────────────────────────────────────────────────────────
def compute_metrics(pred):
    logits, labels = pred
    preds = logits.argmax(axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "micro_f1": f1_score(labels, preds, average="micro"),
        "accuracy": accuracy_score(labels, preds),
    }


# ────────────────────────────────────────────────────────────────────────────────
# TrainingArguments
# ────────────────────────────────────────────────────────────────────────────────
ta_params = dict(
    output_dir=args.out,
    per_device_train_batch_size=micro_batch,
    per_device_eval_batch_size=micro_batch,
    learning_rate=args.lr,
    num_train_epochs=args.epochs,
    logging_steps=50,
    save_strategy="epoch",
    metric_for_best_model="eval_macro_f1",
    load_best_model_at_end=True,
    fp16=is_cuda,          # FP16 only on CUDA
    bf16=False,
    gradient_checkpointing=not is_mps,  # skip on MPS – avoids warnings
    gradient_accumulation_steps=grad_accum,
    report_to="none",
    seed=42,
    optim="adamw_torch",              # never auto‑switch to bnb 8‑bit optim
)

key = "evaluation_strategy"
if key in inspect.signature(TrainingArguments.__init__).parameters:
    ta_params[key] = "epoch"
else:
    ta_params["eval_strategy"] = "epoch"

args_tr = TrainingArguments(**ta_params)


# ────────────────────────────────────────────────────────────────────────────────
# Trainer
# ────────────────────────────────────────────────────────────────────────────────
trainer = Trainer(
    model=model,
    args=args_tr,
    train_dataset=ds_tok["train"],
    eval_dataset=ds_tok["validation"],
    tokenizer=tok,
    compute_metrics=compute_metrics,
)

trainer.train()
val_metrics = trainer.evaluate(ds_tok["validation"])
print("Validation:", val_metrics)
test_metrics = trainer.evaluate(ds_tok["test"])
print("Test:", test_metrics)


# ────────────────────────────────────────────────────────────────────────────────
# Merge LoRA + base and save a single inference‑ready checkpoint
# ────────────────────────────────────────────────────────────────────────────────
trainer.model.merge_and_unload()
merged_dir = os.path.join(args.out, "merged")
trainer.model.save_pretrained(merged_dir)
tok.save_pretrained(merged_dir)
print(f"✅ finished – merged model at {merged_dir}")