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
import pathlib

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
    LANG_DETECTOR,
)


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
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.add_language and not LANG_DETECTOR:
        print("⚠️  Language detector not available; language column will be 'unk'.")
    cluster2cat = load_mapping(args.map)
    labels = sorted(set(cluster2cat.values()))
    print(f"✔ found {len(labels)} categories in mapping file")

    bar_opts = dict(dynamic_ncols=True, leave=False, position=0)
    stream = json_stream(args.zip, cluster2cat, bar_opts, args.add_language)
    ds_full = materialise(stream, labels, args.add_language)
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
