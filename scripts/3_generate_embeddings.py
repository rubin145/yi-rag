#!/usr/bin/env python3
"""
Step 3: Generate dense embeddings for chunks based on `config/vectors.yaml`.
"""
import argparse
import logging
import os
import sys
from pathlib import Path
import yaml
import pandas as pd
from tqdm import tqdm

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.embeddings import embed_texts

def main():
    parser = argparse.ArgumentParser(
        description="Generate dense embeddings for preprocessed chunks based on a YAML config.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--input", "-i",
        default="data/processed/chunks_with_sparse_vectors.parquet",
        help="Input parquet file with chunks and sparse vectors."
    )
    parser.add_argument(
        "--output", "-o", 
        default="data/processed/chunks_with_all_vectors.parquet",
        help="Output parquet file with added dense vectors."
    )
    parser.add_argument(
        "--config",
        default="config/vectors.yaml",
        help="Path to the vector configuration file."
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
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' does not exist.")
        sys.exit(1)

    print("🚀 Starting dense embedding generation (Step 3)...")
    
    df = pd.read_parquet(args.input)
    
    with open(args.config, 'r') as f:
        vector_configs = yaml.safe_load(f).get('vectors', [])

    if not vector_configs:
        print("No vector configurations found in config file. Exiting.")
        sys.exit(0)

    # Generate embeddings for each configuration and store them in temporary columns
    for config in tqdm(vector_configs, desc="Processing vector configs"):
        vec_name = config.get('name')
        source_field = config.get('source_field', 'yi_text')  # Default to yi_text

        if not vec_name:
            print("Skipping a config without 'name'.")
            continue

        # Skip if the source field is empty / missing
        if source_field not in df.columns or df[source_field].isnull().all():
            print(f"Skipping '{vec_name}' because source field '{source_field}' is empty or does not exist.")
            continue

        print(f"Generating embeddings for '{vec_name}' using provider={config.get('provider')} model={config.get('model')} field='{source_field}' ...")

        # Prepare texts
        texts_to_embed = df[source_field].fillna('').tolist()

        # Call functional embedding API; pass full config (embed_texts will coerce spec)
        try:
            generated_vectors = embed_texts(texts_to_embed, config)
        except Exception as e:
            print(f"Error generating embeddings for {vec_name}: {e}")
            df[vec_name] = [None] * len(df)
            continue

        df[vec_name] = generated_vectors

    # Now, consolidate the temporary vector columns into the final 'vectors' dictionary column
    vector_names = [vc['name'] for vc in vector_configs if vc['name'] in df.columns]
    print(f"Consolidating generated vectors: {vector_names}")

    def consolidate_vectors(row):
        vector_dict = {}
        for name in vector_names:
            if name in row and row[name] is not None:
                vector_dict[name] = row[name]
        return vector_dict

    df['vectors'] = df.apply(consolidate_vectors, axis=1)

    # Drop the temporary columns
    df = df.drop(columns=vector_names)

    print(f"Saving DataFrame with all vectors to {args.output}...")
    df.to_parquet(args.output, index=False)
    
    print(f"✅ Dense embeddings created and saved to {args.output}")

if __name__ == "__main__":
    main()
