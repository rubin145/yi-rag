#!/usr/bin/env python3
"""
New script (Step 2): Build and save sparse vectors.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import numpy as np

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.sparse_vectors import CharTfidfSparseEncoder


def main():
    parser = argparse.ArgumentParser(
        description="Build and save sparse vectors for the preprocessed chunks.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python scripts/2_build_sparse_vectors.py 
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        default="data/processed/chunks_untranslated.parquet",
        help="Input parquet file with untranslated chunks."
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="data/processed/chunks_with_sparse_vectors.parquet",
        help="Output parquet file with added sparse vectors."
    )
    
    parser.add_argument(
        "--model-output-dir",
        default="data/sparse_model",
        help="Directory to save the fitted sparse vector model."
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

    print("🚀 Starting sparse vector generation (Step 2)...")
    
    df = pd.read_parquet(args.input)
    
    print("Fitting sparse vector model...")
    encoder = CharTfidfSparseEncoder()
    encoder.fit(df['yi_text'])
    
    print(f"Saving fitted model to {args.model_output_dir}...")
    encoder.save(args.model_output_dir)

    print("Transforming text to sparse vectors...")
    indices_values_list = encoder.transform_to_indices_values(df['yi_text'])
    
    sparse_vectors = [
        {"indices": idx.tolist(), "values": val.tolist()}
        for idx, val in tqdm(indices_values_list, desc="Formatting sparse vectors")
    ]
        
    df['sparse_vector_yi'] = sparse_vectors
    
    # Ensure the column type is object for Parquet compatibility
    df['sparse_vector_yi'] = df['sparse_vector_yi'].astype('object')

    print(f"Saving DataFrame with sparse vectors to {args.output}...")
    df.to_parquet(args.output, index=False)
    
    print(f"✅ Sparse vectors created and saved to {args.output}")


if __name__ == "__main__":
    main()
