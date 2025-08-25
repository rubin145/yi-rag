#!/usr/bin/env python3
"""
Step 6 (optional / repair): Verify and (if needed) add sparse vectors to an existing Qdrant collection.

Use cases:
 1. Collection was originally created without sparse_vectors_config and later recreated (needs full re-index).
 2. Collection already has sparse vector support but points were inserted before sparse vectors were generated.

This script will:
  - Check if the collection supports sparse vectors (abort with instructions if not)
  - Optionally regenerate sparse vectors on the fly using the saved TF-IDF model OR load them from a parquet file
  - Batch-update points adding the "sparse_vector_yi" sparse vector
  - Skip points that already possess the sparse vector (unless --force)

Examples:
  python scripts/6_add_sparse_vectors.py --collection main --model-dir data/sparse_model --batch-size 200
  python scripts/6_add_sparse_vectors.py --collection main --parquet data/processed/chunks_with_all_vectors.parquet --dry-run

Exit codes:
  0 success / nothing to do
  1 fatal error / misconfiguration
"""
import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional

import pandas as pd
from tqdm import tqdm

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from qdrant_client import QdrantClient, models
from core.config import settings
from core.sparse_vectors import CharTfidfSparseEncoder
from core.store_qdrant import point_id_from_chunk_id

log = logging.getLogger("add_sparse_vectors")


def load_sparse_from_parquet(parquet_path: Path) -> Dict[str, Dict[str, List[float]]]:
    df = pd.read_parquet(parquet_path)
    if "sparse_vector_yi" not in df.columns:
        raise ValueError(f"Parquet file {parquet_path} lacks 'sparse_vector_yi' column")
    mapping: Dict[str, Dict[str, List[float]]] = {}
    for row in df.itertuples(index=False):
        row_dict = row._asdict() if hasattr(row, "_asdict") else row._asdict  # type: ignore
        rid = row_dict["id"]
        sv = row_dict.get("sparse_vector_yi")
        if isinstance(sv, dict) and sv.get("indices") and sv.get("values"):
            mapping[rid] = sv
    return mapping


def generate_sparse_vectors(client: QdrantClient, collection: str, model_dir: Path, batch_size: int, limit: Optional[int]) -> Dict[str, Dict[str, List[float]]]:
    encoder = CharTfidfSparseEncoder.load(model_dir)
    log.info("Loaded sparse model from %s", model_dir)

    # Scroll all points requesting only needed payload fields
    result: Dict[str, Dict[str, List[float]]] = {}
    offset = None
    scanned = 0
    while True:
        points, offset = client.scroll(
            collection_name=collection,
            with_payload=True,
            with_vectors=False,
            limit=batch_size,
            offset=offset,
        )
        if not points:
            break
        texts = []
        ids = []
        for p in points:
            payload = p.payload or {}
            if payload.get("yi_text") and isinstance(payload.get("yi_text"), str):
                ids.append(payload.get("id") or payload.get("chunk_id") or payload.get("book_id"))
                # Fallback to reconstruct id if missing: book_id:page:chunk_idx
                if not ids[-1] and all(k in payload for k in ("book_id", "page", "chunk_idx")):
                    ids[-1] = f"{payload['book_id']}:{payload['page']}:{payload['chunk_idx']}"
                texts.append(payload["yi_text"])
        if ids:
            indices_values = encoder.transform_to_indices_values(texts)
            for cid, (idx_arr, val_arr) in zip(ids, indices_values):
                if cid:
                    result[cid] = {"indices": idx_arr.tolist(), "values": val_arr.tolist()}
        scanned += len(points)
        if limit and scanned >= limit:
            break
        if offset is None:
            break
    log.info("Generated sparse vectors for %d chunks", len(result))
    return result


def main():
    parser = argparse.ArgumentParser(description="Verify / add sparse vectors to existing Qdrant collection")
    parser.add_argument("--collection", default=settings.collection_name, help="Collection name")
    parser.add_argument("--parquet", help="Parquet file containing existing sparse vectors (alternative to --model-dir regeneration)")
    parser.add_argument("--model-dir", default="data/sparse_model", help="Directory with saved sparse model (used if --parquet not provided)")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for scrolling / updates")
    parser.add_argument("--limit", type=int, help="Optional limit on points to process (for testing)")
    parser.add_argument("--dry-run", action="store_true", help="Do not perform writes; just report counts")
    parser.add_argument("--force", action="store_true", help="Overwrite existing sparse vectors if present")
    parser.add_argument("--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"])    

    args = parser.parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s [%(name)s] %(message)s")

    client = QdrantClient(url=settings.qdrant_url, api_key=settings.qdrant_api_key, timeout=60)

    # Collection info
    try:
        coll = client.get_collection(args.collection)
    except Exception as e:
        log.error("Failed to fetch collection '%s': %s", args.collection, e)
        sys.exit(1)

    sparse_supported = bool(getattr(coll.config.params, "sparse_vectors", None))
    if not sparse_supported:
        log.error("Collection '%s' has no sparse vector configuration. You must recreate it (Step 4) before adding sparse vectors.")
        log.error("Suggested: python scripts/4_index_corpus.py --input data/processed/chunks_with_all_vectors.parquet --collection-name %s")
        sys.exit(1)

    # Determine source of sparse vectors
    if args.parquet:
        log.info("Loading sparse vectors from parquet: %s", args.parquet)
        sparse_map = load_sparse_from_parquet(Path(args.parquet))
    else:
        log.info("Regenerating sparse vectors from model dir: %s", args.model_dir)
        sparse_map = generate_sparse_vectors(client, args.collection, Path(args.model_dir), args.batch_size, args.limit)

    if not sparse_map:
        log.info("No sparse vectors to process. Exiting.")
        return

    # Identify which points already have sparse vector (if not forcing)
    to_update: List[models.PointVectors] = []
    processed = 0
    updated = 0
    offset = None
    while True:
        points, offset = client.scroll(
            collection_name=args.collection,
            with_payload=True,
            with_vectors=["sparse_vector_yi"],
            limit=args.batch_size,
            offset=offset,
        )
        if not points:
            break
        for p in points:
            payload = p.payload or {}
            cid = payload.get("id") or payload.get("chunk_id")
            if not cid and all(k in payload for k in ("book_id", "page", "chunk_idx")):
                cid = f"{payload['book_id']}:{payload['page']}:{payload['chunk_idx']}"
            if not cid:
                continue
            processed += 1
            if args.limit and processed > args.limit:
                break
            has_sparse = False
            vec_container = getattr(p, "vector", None) or getattr(p, "vectors", None)
            if isinstance(vec_container, dict) and "sparse_vector_yi" in vec_container:
                has_sparse = True
            if (not has_sparse or args.force) and cid in sparse_map:
                sv = sparse_map[cid]
                to_update.append(
                    models.PointVectors(
                        id=point_id_from_chunk_id(cid),
                        vector={
                            "sparse_vector_yi": models.SparseVector(
                                indices=sv["indices"], values=sv["values"]
                            )
                        },
                    )
                )
                updated += 1
        if args.limit and processed >= (args.limit or 0):
            break
        if offset is None:
            break

    log.info("Scanned points: %d", processed)
    log.info("Points needing update: %d", updated)

    if args.dry_run:
        log.info("Dry run enabled; not sending updates.")
        return

    if not to_update:
        log.info("Nothing to update.")
        return

    for i in range(0, len(to_update), args.batch_size):
        batch = to_update[i : i + args.batch_size]
        client.update_vectors(collection_name=args.collection, points=batch, wait=True)
    log.info("Sparse vector updates complete (%d points updated)", updated)


if __name__ == "__main__":
    main()
