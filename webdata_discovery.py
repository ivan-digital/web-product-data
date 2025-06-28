#!/usr/bin/env python3
"""
webdata_discovery.py
====================
Language / brand / manufacturer / **24‑class category** profiler
for the Web‑Data‑Commons Large‑Scale Product Corpus V2020
(robust against URL moves & 403/404 errors).
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import csv
import functools
import gzip
import io
import json
import logging
import os
import pathlib
import re
import sys
import zipfile
from typing import Any, Counter, DefaultDict, Dict, List, Tuple

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─────────────────────────  logging  ──────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(levelname)s:%(name)s:%(message)s",
                    stream=sys.stderr)
log = logging.getLogger("lspc2020")

# ───────────  fast or fallback language detector  ─────────────
try:
    from ftlangdetect import detect as ft_detect           # pip install ftlangdetect

    def autodetect(text: str) -> str:
        text = re.sub(r"\s+", " ", text).strip()
        if not text:
            return "unk"
        try:
            res = ft_detect(text)
            return res["lang"] if isinstance(res, dict) else res.lang
        except Exception:
            return "unk"
except ModuleNotFoundError:                                   # fallback → langdetect
    log.info("fasttext‑langdetect not installed – using langdetect")
    from langdetect import detect as slow_detect              # type: ignore

    def autodetect(text: str) -> str:
        if not text.strip():
            return "unk"
        try:
            return slow_detect(text)
        except Exception:
            return "unk"

# ─────────────────────────  constants  ─────────────────────────
INDEX_URL = ("https://webdatacommons.org/"
             "largescaleproductcorpus/v2020/index.html")

# fallback JSON (6 labels) – used only if every CSV mirror fails
MAJ_VOTE_JSON = (
    "https://data.dws.informatik.uni-mannheim.de/"
    "largescaleproductcorpus/data/wdc-products/categorization/"
    "WDC_Corpus_LargeScaleExperiment_MajorityVoting.json.gz"
)

# official 24 flat labels
_LABELS = {
    'Automotive', 'Baby', 'Beauty & Personal Care', 'Books',
    'Cell Phones & Accessories', 'Clothing', 'Computers & Accessories',
    'Electronics', 'Grocery & Gourmet Food', 'Health & Household',
    'Home & Garden', 'Industrial & Scientific', 'Movies & TV',
    'Music', 'Office Products', 'Pet Supplies', 'Shoes',
    'Sports & Outdoors', 'Tools & Home Improvement', 'Toys & Games',
    'Video Games', 'Jewelry', 'Luggage', 'Handmade'
}

# ────────────────────────  helpers  ────────────────────────────
def _open_maybe_gzip(path: pathlib.Path, mode: str = "rt"):
    """Return a text handle that works for both gzip and plain files."""
    with path.open("rb") as probe:
        magic = probe.read(2)
    return gzip.open(path, mode, encoding="utf-8") if magic == b"\x1f\x8b" \
        else path.open(mode, encoding="utf-8")


def _json_to_cat(rec: dict) -> str | None:
    """Depth‑first walk to find the first string that matches one of the 24 labels."""
    stack: List[Any] = [rec]
    while stack:
        node = stack.pop()
        if isinstance(node, dict):
            for v in node.values():
                if isinstance(v, str) and v in _LABELS:
                    return v
                stack.append(v)
        elif isinstance(node, list):
            stack.extend(node)
    return None

# ───────────────────  corpus ZIP (auto‑download)  ─────────────────
def ensure_zip(local: pathlib.Path | None) -> pathlib.Path:
    if local:
        if not local.exists():
            raise FileNotFoundError(local)
        return local

    path = pathlib.Path("lspcV2020.zip")
    if path.exists():
        return path

    log.info("⇣ downloading corpus index …")
    page = BeautifulSoup(requests.get(INDEX_URL, timeout=30).text, "html.parser")
    href = next((a["href"] for a in page.select('a[href$=".zip"]') if "V2020" in a["href"]),
                None)
    if not href:
        raise RuntimeError("ZIP link not found on index page")
    url = requests.compat.urljoin(INDEX_URL, href)
    log.info("⇣ downloading %s … (~20 GB)", url.split("/")[-1])

    with requests.get(url, stream=True, timeout=60) as r, path.open("wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)
    log.info("ZIP saved → %s", path)
    return path

# ───────────────  category mapping (robust)  ───────────────────────
def ensure_mapping(tsv_path: pathlib.Path) -> pathlib.Path:
    """Download (or reuse) cluster_id<TAB>category mapping with 24 labels."""
    if tsv_path.exists() and tsv_path.stat().st_size:
        return tsv_path                                    # already cached

    tsv_path.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0",
               "Referer": "https://webdatacommons.org/"}

    # ---------- last‑ditch fallback ----------
    tmp = tsv_path.with_suffix(".json.gz")
    log.info("⇣ downloading majority‑voting JSON …")
    with requests.get(MAJ_VOTE_JSON, headers=headers,
                      stream=True, timeout=60) as r, tmp.open("wb") as fh:
        for chunk in r.iter_content(1 << 20):
            fh.write(chunk)

    log.info("↻ converting JSON → TSV …")
    written = 0
    with _open_maybe_gzip(tmp) as jin, tsv_path.open("w", newline='', encoding="utf-8") as jout:
        w = csv.writer(jout, delimiter='\t')
        for line in jin:
            rec = json.loads(line)
            if (cat := _json_to_cat(rec)):
                w.writerow([rec["cluster_id"], cat])
                written += 1
    tmp.unlink(missing_ok=True)
    log.info("Converted %s records → %s", written, tsv_path)
    return tsv_path

# ─────────────  mapping loader (cached per process)  ──────────────
@functools.lru_cache(maxsize=1)
def load_map(path: str) -> Dict[int, str]:
    m: Dict[int, str] = {}
    with pathlib.Path(path).open(newline='', encoding="utf-8") as fh:
        for cid, cat in csv.reader(fh, delimiter='\t'):
            try:
                m[int(cid)] = cat
            except ValueError:
                continue
    return m

# ───────────────────────  worker  ────────────────────────────────
def scan_member(args: Tuple[str, str, int, int, str]) -> Tuple[
    Counter[str],
    Dict[str, List[Tuple[str, str]]],
    Dict[str, Counter[str]],
    Dict[str, Counter[str]],
    Dict[str, Counter[str]],
]:
    zip_path, member, keep, limit, map_path = args
    cat_map = load_map(map_path)

    counts: Counter[str] = collections.Counter()
    samples: DefaultDict[str, List[Tuple[str, str]]] = collections.defaultdict(list)
    brands:  DefaultDict[str, Counter[str]] = collections.defaultdict(collections.Counter)
    manufs:  DefaultDict[str, Counter[str]] = collections.defaultdict(collections.Counter)
    cats:    DefaultDict[str, Counter[str]] = collections.defaultdict(collections.Counter)

    processed = 0
    with zipfile.ZipFile(zip_path) as z, \
         z.open(member) as raw, gzip.GzipFile(fileobj=raw) as gz, \
         io.TextIOWrapper(gz, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            if limit and processed >= limit:
                break
            processed += 1
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = " ".join(filter(None, (obj.get("title"),
                                          obj.get("name"),
                                          obj.get("description"))))
            lang = obj.get("language") or autodetect(text)
            counts[lang] += 1

            if len(samples[lang]) < keep:
                samples[lang].append(((obj.get("title") or obj.get("name") or "")[:120],
                                      (obj.get("url") or "")[:80]))

            for fld, bucket in (("brand", brands), ("manufacturer", manufs)):
                if (val := obj.get(fld)):
                    bucket[lang][str(val).lower()] += 1

            if (cid := obj.get("cluster_id")) is not None:
                if (cat := cat_map.get(int(cid))):
                    cats[lang][cat] += 1

    return counts, samples, brands, manufs, cats

# ────────────────────  merge utilities  ───────────────────────────
def _merge_ctr(dst: Counter[Any], src: Counter[Any]) -> None:
    for k, v in src.items():
        dst[k] += v


def _merge_nested(dst: Dict[str, Counter[str]], src: Dict[str, Counter[str]]) -> None:
    for lang, ctr in src.items():
        _merge_ctr(dst.setdefault(lang, collections.Counter()), ctr)


def _merge_samples(dst: Dict[str, List[Tuple[str, str]]],
                   src: Dict[str, List[Tuple[str, str]]], keep: int) -> None:
    for lang, rows in src.items():
        need = keep - len(dst.setdefault(lang, []))
        if need > 0:
            dst[lang].extend(rows[:need])

# ─────────────────────  reporting helpers  ────────────────────────
def show_counts(counts: Counter[str]) -> None:
    tot = sum(counts.values())
    print("\nLanguage  Offers        Share")
    print("--------- ------------- ------")
    for lang, n in counts.most_common():
        print(f"{lang:<8} {n:>13,} {n/tot:5.1%}")
    print("--------- -------------")
    print(f"TOTAL     {tot:,}")


def show_top(nested: Dict[str, Counter[str]], label: str, n=10) -> None:
    print(f"\n=== Top {n} {label}s per language ===")
    for lang, ctr in nested.items():
        if ctr:
            head = ", ".join(f"{t} ({f})" for t, f in ctr.most_common(n))
            print(f"▶ {lang}: {head}")


def show_cats(cats: Dict[str, Counter[str]], n=10) -> None:
    if not cats:
        return
    print("\n=== Category distribution per language ===")
    for lang, ctr in cats.items():
        tot = sum(ctr.values())
        head = ", ".join(f"{c} ({v/tot:.1%})" for c, v in ctr.most_common(n))
        print(f"▶ {lang:<8} {tot:>10,}  {head}")

# ───────────────────────────  main  ───────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser("LSPC V2020 language + category profiler")
    ap.add_argument("--zip", type=pathlib.Path,
                    help="Use an existing lspcV2020.zip")
    ap.add_argument("--map", type=pathlib.Path,
                    default=pathlib.Path("PDC2020_map.tsv"),
                    help="Cluster→category TSV cache (auto‑created)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(),
                    help="Processes to spawn [all logical CPUs]")
    ap.add_argument("--limit", type=int, default=0,
                    help="Offers per worker (0 = all)")
    ap.add_argument("--keep", type=int, default=3,
                    help="Sample titles kept per language [3]")
    args = ap.parse_args()

    zip_path = ensure_zip(args.zip)
    map_path = ensure_mapping(args.map)

    with zipfile.ZipFile(zip_path) as z:
        members = sorted(m for m in z.namelist() if m.endswith(".json.gz"))

    g_counts: Counter[str] = collections.Counter()
    g_samples: Dict[str, List[Tuple[str, str]]] = {}
    g_brands: Dict[str, Counter[str]] = {}
    g_manufs: Dict[str, Counter[str]] = {}
    g_cats:   Dict[str, Counter[str]] = {}

    tasks = [(str(zip_path), m, args.keep, args.limit or 0, str(map_path))
             for m in members]

    log.info("Launching %d workers …", args.workers)
    bar_opts = dict(dynamic_ncols=True, leave=False, position=0)
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as ex:
        for part in tqdm(ex.map(scan_member, tasks), total=len(tasks),
                         desc="merge", **bar_opts):
            cnt, samp, br, mf, ct = part
            _merge_ctr(g_counts, cnt)
            _merge_samples(g_samples, samp, args.keep)
            _merge_nested(g_brands, br)
            _merge_nested(g_manufs, mf)
            _merge_nested(g_cats, ct)

    show_counts(g_counts)
    show_top(g_brands,  "brand")
    show_top(g_manufs,  "manufacturer")
    show_cats(g_cats)


if __name__ == "__main__":
    main()