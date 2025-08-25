#!/usr/bin/env python3
"""Utility: Fix already indexed Qdrant points metadata.

Repairs:
 1. Restore correct title_yi_latin from corpus_metadata.csv (CSV 'title' column) if missing or incorrect.
 2. Populate ocr_source_url and metadata_url if empty using corpus_metadata.csv.
  3. Ensure tr_en_metadata dict exists with provider/model/system_prompt when tr_en_text present.

Safe (idempotent) – only updates missing / incorrect fields.
"""
import sys, os, logging, json
from pathlib import Path
from typing import Dict
import pandas as pd

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.config import settings  # noqa: E402
from core.store_qdrant import QdrantStore  # noqa: E402
from core.translate import Translator  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger("fix_metadata")

CSV_PATH = "data/corpus_metadata.csv"


def load_csv() -> Dict[str, Dict]:
    if not os.path.exists(CSV_PATH):
        raise SystemExit(f"Metadata CSV not found: {CSV_PATH}")
    df = pd.read_csv(CSV_PATH).fillna("")
    out = {}
    for _, r in df.iterrows():
        bid = str(r.get("book_id", "")).strip()
        if not bid:
            continue
        out[bid] = {
            "title_yi_latin": (str(r.get("title", "")).strip() or None),
            "title_yi": (str(r.get("title_yi", "")).strip() or None),
            "ocr_source_url": (str(r.get("ocr_source_url", "")).strip() or None),
            "metadata_url": (str(r.get("metadata_url", "")).strip() or None),
        }
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Fix indexed metadata fields in Qdrant")
    ap.add_argument("--collection", default=settings.collection_name)
    ap.add_argument("--limit", type=int, default=None, help="Process only first N points")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    store = QdrantStore(args.collection)
    meta_csv = load_csv()
    translator = Translator()  # for system prompt / provider info only

    updated = 0
    scanned = 0

    next_page = None
    while True:
        points, next_page = store.client.scroll(
            collection_name=store.collection_name,
            with_payload=True,
            with_vectors=False,
            limit=500,
            offset=next_page,
        )
        if not points:
            break
        for p in points:
            if args.limit and scanned >= args.limit:
                break
            scanned += 1
            payload = p.payload or {}
            book_id = payload.get("book_id")
            if not book_id:
                continue
            csv_row = meta_csv.get(book_id, {})
            patch = {}
            if csv_row.get("title_yi_latin") and payload.get("title_yi_latin") != csv_row.get("title_yi_latin"):
                patch["title_yi_latin"] = csv_row["title_yi_latin"]
            for fld in ("ocr_source_url", "metadata_url"):
                if csv_row.get(fld) and not payload.get(fld):
                    patch[fld] = csv_row[fld]
            if payload.get("tr_en_text") and not payload.get("tr_en_metadata"):
                patch["tr_en_metadata"] = {
                    "provider": translator.provider,
                    "model": translator.model,
                    "system_prompt": translator.get_system_prompt(),
                }
            if patch:
                log.info("Updating point %s patch=%s", p.id, json.dumps(patch, ensure_ascii=False))
                if not args.dry_run:
                    store.client.set_payload(
                        collection_name=store.collection_name,
                        payload=patch,
                        points=[p.id],
                        wait=False,
                    )
                updated += 1
        if args.limit and scanned >= args.limit:
            break
        if next_page is None:
            break
    log.info("Done. scanned=%d updated=%d", scanned, updated)


if __name__ == "__main__":
    main()
