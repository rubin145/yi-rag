#!/usr/bin/env python
"""Quick diagnostic script to compare dense (semantic) vs sparse (lexical TF‑IDF) retrieval.

Usage examples:
  poetry run python scripts/diagnose_retrieval.py --query "Bund Warsaw strikes" \
      --k 8 --vector dense_yi_cohere_1536
  poetry run python scripts/diagnose_retrieval.py --queries "Bund Warsaw" "organizational experience" --k 10

It prints:
  * Top-k ids & scores for dense and sparse
  * Overlap counts / Jaccard similarity
  * Whether scores or ids are identical (suspicious for sparse path)
  * Per-hit mode tags (expects 'dense' or 'sparse')
  * Optional raw point check to ensure sparse vector field exists

Exit code 0 even if anomalies found; anomalies are flagged with !!! so you can grep.
"""
from __future__ import annotations

import os
import sys
import argparse
from typing import List, Dict, Any

 # Ensure project root (parent of scripts/) on path for local execution
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from core.retrieve import Retriever, SPARSE_VECTOR_NAME  # type: ignore
from core.store_qdrant import get_client  # type: ignore


def format_hits(hits: List[Dict[str, Any]]) -> List[str]:
    return [f"{i+1}:{h.get('id')}({h.get('score'):.4f},{h.get('mode')})" for i, h in enumerate(hits)]


def jaccard(a: List[Any], b: List[Any]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0


def run_query(r: Retriever, query: str, k: int, vector: str):
    dense_hits = r.search(query=query, k=k, using=vector, with_payload=False)
    sparse_hits = r.sparse_search(query=query, k=k, with_payload=False)

    dense_ids = [h['id'] for h in dense_hits]
    sparse_ids = [h['id'] for h in sparse_hits]
    overlap = len(set(dense_ids) & set(sparse_ids))
    jac = jaccard(dense_ids, sparse_ids)
    identical_ids = dense_ids == sparse_ids
    identical_scores = all(abs(dense_hits[i]['score'] - sparse_hits[i]['score']) < 1e-9 for i in range(min(len(dense_hits), len(sparse_hits)))) if dense_hits and sparse_hits else False

    print(f"\n=== Query: {query!r} ===")
    print(f"Dense vector: {vector}")
    print("Dense hits:", ", ".join(format_hits(dense_hits)))
    print("Sparse hits:", ", ".join(format_hits(sparse_hits)))
    print(f"Overlap count: {overlap}/{k}  Jaccard: {jac:.3f}")
    if identical_ids:
        print("!!! IDs are identical (unexpected for most lexical vs semantic queries)")
    if identical_scores:
        print("!!! Scores are numerically identical (suggests sparse path fallback)")
    # Show top differing hit
    for i,(d,s) in enumerate(zip(dense_hits, sparse_hits)):
        if d['id'] != s['id']:
            print(f"First rank where they diverge: position {i+1}: dense={d['id']} sparse={s['id']}")
            break


def verify_sparse_presence(r: Retriever, n: int = 3):
    client = get_client()
    # Scroll a few points to check that sparse vector is retrievable
    print("\nChecking sparse vector presence on a few points...")
    scrolled, _ = client.scroll(collection_name=r.collection, with_payload=False, with_vectors=True, limit=n)
    missing = 0
    for p in scrolled:
        has_sparse = SPARSE_VECTOR_NAME in (p.vector or {})
        print(f"Point {p.id}: has_sparse={has_sparse}")
        if not has_sparse:
            missing += 1
    if missing:
        print(f"!!! {missing}/{len(scrolled)} sample points missing sparse vectors")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--query', help='Single query')
    ap.add_argument('--queries', nargs='*', help='Multiple queries')
    ap.add_argument('--k', type=int, default=8)
    ap.add_argument('--vector', default=None, help='Named dense vector to use (default: first containing yi)')
    ap.add_argument('--skip-sparse-check', action='store_true')
    args = ap.parse_args()

    r = Retriever(vectors_yaml='config/vectors.yaml')
    if args.vector is None:
        # choose a yi vector first else first vector
        yi_vecs = [v for v in r.list_vectors() if 'yi' in v]
        vector = yi_vecs[0] if yi_vecs else r.list_vectors()[0]
    else:
        vector = args.vector

    queries = []
    if args.query:
        queries.append(args.query)
    if args.queries:
        queries.extend(args.queries)
    if not queries:
        queries = ["Warsaw Bund strike", "organizational experience literature", "translation agitation"]

    print(f"Using dense vector: {vector}")
    for q in queries:
        run_query(r, q, args.k, vector)

    if not args.skip_sparse_check:
        verify_sparse_presence(r)

    print("\nDone. Investigate any !!! lines above.")


if __name__ == '__main__':
    main()
