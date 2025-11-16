#!/usr/bin/env python3
"""
build_lspc_dataset_full.py
-------------------------
Variant of `build_lspc_dataset.py` that stores the entire LSPC corpus as an
additional split (default: "full"), making it easy to train on all examples
while still keeping leak-free train/validation/test partitions.

Example:
    python category_classification/build_lspc_dataset_full.py \\
        --zip lspcV2020.zip --map PDC2020_map.tsv --out lspc_dataset_full
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import multiprocessing as mp
import pathlib
import zipfile

# Allow running as a script from this directory or project root
import os
import sys
ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from datasets import DatasetDict

from category_classification.build_lspc_dataset import (
    grouped_split,
    json_stream,
    load_mapping,
    materialise,
    print_stats,
)
from category_classification.language_detection import annotate_dataset, ensure_language_detector


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--zip", type=pathlib.Path, default="lspcV2020.zip",
                   help="Path to lspcV2020.zip")
    p.add_argument("--map", type=pathlib.Path, default="PDC2020_map.tsv",
                   help="Path to PDC2020_map.tsv (cluster_id<TAB>category)")
    p.add_argument("--out", type=pathlib.Path, default="lspc_dataset_full",
                   help="Output directory for saved DatasetDict")
    p.add_argument("--val_size", type=float, default=0.05, help="Fraction for validation set")
    p.add_argument("--test_size", type=float, default=0.10, help="Fraction for test set")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--full_split_name", type=str, default="full",
                   help="Name of the all-data split to add")
    p.add_argument("--full_only", action="store_true",
                   help="If set, only save the full split (no train/val/test)")
    p.add_argument("--add_language", action="store_true",
                   help="Annotate each record with a detected language code")
    p.add_argument("--language_num_proc", type=int, default=None,
                   help="Worker processes for language detection (default: half of CPU cores)")
    p.add_argument("--read_workers", type=int, default=1,
                   help="Number of worker processes for scanning JSON files (default: 1)")
    return p.parse_args()


def _process_member(args):
    zip_path, member, cluster2cat = args
    items = []
    with zipfile.ZipFile(zip_path) as z:
        with z.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz, \
                io.TextIOWrapper(gz, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                rec = json.loads(line)
                cat = cluster2cat.get(rec.get("cluster_id"))
                if not cat:
                    continue
                text = " ".join(filter(None, (
                    rec.get("title") or rec.get("name"),
                    rec.get("description") or ""
                )))
                if text.strip():
                    items.append({
                        "text": text,
                        "label": cat,
                        "cluster": int(rec["cluster_id"]),
                    })
    return items


def parallel_json_stream(zip_path: pathlib.Path,
                         cluster2cat: dict[int, str],
                         workers: int):
    with zipfile.ZipFile(zip_path) as z:
        members = [m for m in z.namelist() if m.endswith(".json.gz")]
    args_iter = ((zip_path, member, cluster2cat) for member in members)
    with mp.Pool(processes=workers) as pool:
        for batch in pool.imap_unordered(_process_member, args_iter, chunksize=1):
            for record in batch:
                yield record


def main() -> None:
    args = parse_args()
    if args.add_language:
        print("⚙️  Language detection enabled (fasttext preferred).")
    cluster2cat = load_mapping(args.map)
    labels = sorted(set(cluster2cat.values()))
    print(f"✔ found {len(labels)} categories in mapping file")

    if args.read_workers and args.read_workers > 1:
        stream = parallel_json_stream(args.zip, cluster2cat, args.read_workers)
        ds_full = materialise(stream, labels, False)
    else:
        bar_opts = dict(dynamic_ncols=True, leave=False, position=0)
        stream = json_stream(args.zip, cluster2cat, bar_opts, False)
        ds_full = materialise(stream, labels, False)
    if args.add_language:
        ensure_language_detector()
        ds_full = annotate_dataset(ds_full, num_proc=args.language_num_proc)
    print(f"✔ corpus materialised: {len(ds_full):,} records")

    ds_entries: dict[str, object] = {args.full_split_name: ds_full}

    if not args.full_only:
        train_ds, val_ds, test_ds = grouped_split(ds_full,
                                                  test_size=args.test_size,
                                                  val_size=args.val_size,
                                                  seed=args.seed)
        for name, d in [("train", train_ds), ("validation", val_ds), ("test", test_ds)]:
            print_stats(name, d, args.add_language and "language" in d.column_names)
            ds_entries[name] = d
    else:
        print("⚠️  --full_only specified; skipping train/validation/test splits.")

    dsdict = DatasetDict(ds_entries)
    args.out.mkdir(parents=True, exist_ok=True)
    dsdict.save_to_disk(args.out)
    print(f"\n✅ DatasetDict saved to: {args.out.resolve()} (splits: {list(ds_entries)})")


if __name__ == "__main__":
    main()
