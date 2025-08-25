#!/usr/bin/env python3
"""
New script (Step 4): Index corpus with sparse and dense vectors.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.store_qdrant import QdrantStore
from core.schemas import Chunk
from core.config import settings

def main():
    parser = argparse.ArgumentParser(
        description="Index preprocessed chunks with dense and sparse vectors into Qdrant.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--input", "-i",
        default="data/processed/chunks_with_all_vectors.parquet",
        help="Input parquet file with chunks and all vectors."
    )
    
    parser.add_argument(
        "--collection-name",
        default=settings.collection_name,
        help="Name of the Qdrant collection (default from settings)."
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level."
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        sys.exit(1)

    print("🚀 Starting indexing (Step 4)...")
    
    df = pd.read_parquet(args.input)
    
    qdrant_store = QdrantStore(args.collection_name)
    
    qdrant_store.recreate_collection()

    chunks_to_index = [Chunk(**row) for row in df.to_dict(orient='records')]
    
    qdrant_store.index_chunks(chunks_to_index)

    print(f"✅ Indexing complete for collection '{args.collection_name}'.")


if __name__ == "__main__":
    main()
