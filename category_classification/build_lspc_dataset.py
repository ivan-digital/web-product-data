#!/usr/bin/env python3
"""
build_lspc_dataset.py
--------------------
Materialises the LSPC‑2020 corpus into a Hugging‑Face DatasetDict
with train / validation / test splits suitable for product‑
classification tasks.

Input
-----
* lspcV2020.zip        – official ZIP from the challenge, containing
                         many *.json.gz files (one JSON per line)
* PDC2020_map.tsv      – two‑column TSV mapping cluster_id → category

Output
------
<out>/
  dataset_info.json
  train/  validation/  test/   # Arrow files + metadata

The resulting DatasetDict has three columns:
  • text    (string)           – concatenated title + description
  • label   (ClassLabel)       – 24 product categories
  • cluster (int32)            – original cluster_id

Splitting strategy
------------------
To avoid information leakage, splitting is **grouped by cluster_id**:
all items with the same cluster land in the same split.  Within that
constraint, splits are approximately stratified by label.  Randomness
is controlled by a single seed (default 42) to ensure reproducibility.
"""

from __future__ import annotations
import argparse, csv, gzip, io, json, pathlib, random, zipfile
from collections import Counter
from typing import Dict, Iterable, List, Tuple

import tqdm
from datasets import (Dataset, DatasetDict, Features, Value, ClassLabel,
                      disable_progress_bar, concatenate_datasets)
from sklearn.model_selection import GroupShuffleSplit

disable_progress_bar()  # keep HF quiet

from webdata_discovery import autodetect as detect_language  # type: ignore
LANG_DETECTOR = "autodetect"

# ----------------------------- CLI ----------------------------- #
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--zip",       type=pathlib.Path, default="lspcV2020.zip",
                   help="Path to lspcV2020.zip")
    p.add_argument("--map",       type=pathlib.Path, default="PDC2020_map.tsv",
                   help="Path to PDC2020_map.tsv (cluster_id<TAB>category)")
    p.add_argument("--out",       type=pathlib.Path, default="lspc_dataset",
                   help="Output directory for saved DatasetDict")
    p.add_argument("--val_size",  type=float, default=0.05, help="Fraction for validation set")
    p.add_argument("--test_size", type=float, default=0.10, help="Fraction for test set")
    p.add_argument("--seed",      type=int,   default=42,   help="Random seed")
    p.add_argument("--add_language", action="store_true",
                   help="Annotate each record with a detected language code")
    return p.parse_args()

# --------------------------- Helpers --------------------------- #
def load_mapping(path: pathlib.Path) -> Dict[int, str]:
    """Read TSV cluster→category map (two columns, tab‑separated)."""
    cluster2cat: Dict[int, str] = {}
    with path.open() as fh:
        for cid, cat in csv.reader(fh, delimiter='\t'):
            cluster2cat[int(cid)] = cat.strip()
    if not cluster2cat:
        raise ValueError("⚠️ mapping file is empty or malformed")
    return cluster2cat

def json_stream(zip_path: pathlib.Path,
                cluster2cat: Dict[int, str],
                bar_opts: dict,
                add_language: bool) -> Iterable[dict]:
    """Yield cleaned records one by one from the ZIP archive."""
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".json.gz")]
        for member in tqdm.tqdm(members, desc="scan‑files", **bar_opts):
            with z.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz, \
                 io.TextIOWrapper(gz, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    rec = json.loads(line)
                    cat = cluster2cat.get(rec.get("cluster_id"))
                    if not cat:                         # skip clusters w/o mapping
                        continue
                    text = " ".join(filter(None, (
                        rec.get("title") or rec.get("name"),
                        rec.get("description") or ""
                    )))
                    if text.strip():
                        item = {"text": text,
                                "label": cat,
                                "cluster": int(rec["cluster_id"])}
                        if add_language:
                            item["language"] = detect_language(text) if LANG_DETECTOR else "unk"
                        yield item

def materialise(stream: Iterable[dict],
                labels: List[str],
                add_language: bool) -> Dataset:
    """Turn an iterable of dicts into an in‑memory HF Dataset."""
    feats = Features({
        "text":    Value("string"),
        "label":   ClassLabel(names=labels),
        "cluster": Value("int32"),
    })
    if add_language:
        feats["language"] = Value("string")
    # Since we don’t know the total length in advance, collect into shards
    shard, shards = [], []
    for rec in stream:
        shard.append(rec)
        if len(shard) == 50_000:
            shards.append(Dataset.from_list(shard, features=feats))
            shard = []
    if shard:
        shards.append(Dataset.from_list(shard, features=feats))
    ds = concatenate_datasets(shards) if len(shards) > 1 else shards[0]
    return ds


def grouped_split(ds: Dataset,
                  test_size: float,
                  val_size: float,
                  seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    """Perform group‑aware train/val/test split."""
    # First split off test
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_idx, test_idx = next(gss.split(ds, groups=ds["cluster"]))
    ds_train_val = ds.select(train_idx)
    ds_test       = ds.select(test_idx)

    # Then split train_val→train / val
    val_size_adj = val_size / (1.0 - test_size)  # relative within train_val
    gss_val = GroupShuffleSplit(n_splits=1, test_size=val_size_adj, random_state=seed)
    train_idx, val_idx = next(gss_val.split(ds_train_val,
                                            groups=ds_train_val["cluster"]))
    ds_train = ds_train_val.select(train_idx)
    ds_val   = ds_train_val.select(val_idx)
    return ds_train, ds_val, ds_test

def print_stats(split: str, ds: Dataset, add_language: bool):
    cnt = Counter(ds["label"])
    n   = len(ds)
    top = ", ".join(f"{l}:{c}" for l, c in cnt.most_common(5))
    lang_info = ""
    if add_language and "language" in ds.column_names:
        lang_cnt = Counter(ds["language"])
        lang_top = ", ".join(f"{l}:{c}" for l, c in lang_cnt.most_common(5))
        lang_info = f" • lang_top5: {lang_top}"
    print(f"{split:>9}  {n:>7,} samples   • top5: {top}{lang_info}")

# ----------------------------- main ---------------------------- #
def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    if args.add_language and not LANG_DETECTOR:
        print("⚠️  Language detector not available; language column will be 'unk'.")

    # ---- load mapping & labels ---- #
    cluster2cat = load_mapping(args.map)
    labels = sorted(set(cluster2cat.values()))
    print(f"✔ found {len(labels)} categories in mapping file")

    # ---- stream raw corpus ---- #
    bar_opts = dict(dynamic_ncols=True, leave=False, position=0)
    stream   = json_stream(args.zip, cluster2cat, bar_opts, args.add_language)
    ds_full  = materialise(stream, labels, args.add_language)
    print(f"✔ corpus materialised: {len(ds_full):,} records")

    # ---- leak‑free splits ---- #
    train_ds, val_ds, test_ds = grouped_split(ds_full,
                                              test_size=args.test_size,
                                              val_size=args.val_size,
                                              seed=args.seed)
    for name, d in [("train", train_ds),
                    ("valid", val_ds),
                    ("test",  test_ds)]:
        print_stats(name, d, args.add_language and "language" in d.column_names)

    # ---- save ---- #
    dsdict = DatasetDict({"train": train_ds,
                          "validation": val_ds,
                          "test": test_ds})
    out_path = args.out
    out_path.mkdir(parents=True, exist_ok=True)
    dsdict.save_to_disk(out_path)
    print(f"\n✅ DatasetDict saved to: {out_path.resolve()}")

if __name__ == "__main__":
    main()
