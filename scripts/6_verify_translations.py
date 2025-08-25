#!/usr/bin/env python3
"""Utility: Verify translation coverage in Qdrant collection.

Counts total points and how many have has_translation=1 (or tr_en_text present).
Optionally lists a few missing IDs and can enforce an expected minimum coverage.

Usage examples:
  poetry run python scripts/6_verify_translations.py
  poetry run python scripts/6_verify_translations.py --collection yiddish-corpus --sample-missing 5
  poetry run python scripts/6_verify_translations.py --expect-min-ratio 0.95 --fail-on-low
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root for module imports
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.config import settings  # noqa: E402
from core.store_qdrant import QdrantStore  # noqa: E402
from qdrant_client import models as qm  # noqa: E402


def count_points(store: QdrantStore) -> Tuple[int, int]:
    client = store.client
    total = 0
    translated = 0
    offset = None
    limit = 10_000
    while True:
        points, offset = client.scroll(
            collection_name=store.collection_name,
            with_payload=["has_translation", "tr_en_text"],
            with_vectors=False,
            limit=limit,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            total += 1
            pl = p.payload or {}
            # prefer explicit flag; fallback to presence of tr_en_text
            if pl.get("has_translation") == 1 or (pl.get("tr_en_text") and str(pl.get("tr_en_text")).strip()):
                translated += 1
        if offset is None:
            break
    return total, translated


def sample_missing(store: QdrantStore, n: int) -> List[int]:
    client = store.client
    missing: List[int] = []
    offset = None
    limit = 10_000
    while missing.__len__() < n:
        points, offset = client.scroll(
            collection_name=store.collection_name,
            with_payload=["has_translation", "tr_en_text"],
            with_vectors=False,
            limit=limit,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            if len(missing) >= n:
                break
            pl = p.payload or {}
            if not (pl.get("has_translation") == 1 or (pl.get("tr_en_text") and str(pl.get("tr_en_text")).strip())):
                missing.append(p.id)
        if offset is None:
            break
    return missing


def main():
    ap = argparse.ArgumentParser(description="Verify translation coverage.")
    ap.add_argument("--collection", default=settings.collection_name, help="Qdrant collection name")
    ap.add_argument("--sample-missing", type=int, default=0, help="Show up to N example point IDs missing translation")
    ap.add_argument("--expect-min-ratio", type=float, default=None, help="Expected minimum translated/total ratio (0-1)")
    ap.add_argument("--fail-on-low", action="store_true", help="Exit with code 1 if coverage below expected ratio")
    args = ap.parse_args()

    store = QdrantStore(args.collection)

    total, translated = count_points(store)
    ratio = (translated / total) if total else 0.0

    print(f"Collection: {args.collection}")
    print(f"Total points: {total}")
    print(f"Translated (has_translation=1 or tr_en_text present): {translated}")
    print(f"Coverage: {ratio:.4f}")

    if args.sample_missing > 0:
        missing_ids = sample_missing(store, args.sample_missing)
        if missing_ids:
            print(f"Missing sample ({len(missing_ids)}): {missing_ids}")
        else:
            print("No missing samples found.")

    if args.expect_min_ratio is not None:
        if ratio < args.expect_min_ratio:
            msg = f"Coverage {ratio:.4f} below expected {args.expect_min_ratio:.4f}"
            if args.fail_on_low:
                print("ERROR: " + msg)
                sys.exit(1)
            else:
                print("WARN: " + msg)
        else:
            print("OK: coverage meets expectation")


if __name__ == "__main__":
    main()
