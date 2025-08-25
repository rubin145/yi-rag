#!/usr/bin/env python3
"""
Inspects a Parquet file and prints its schema and a sample of the data.
"""
import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser(
        description="Inspect a Parquet file.",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "file_path",
        help="The path to the Parquet file to inspect."
    )
    
    parser.add_argument(
        "--sample-size",
        type=int,
        default=5,
        help="The number of rows to display as a sample."
    )

    args = parser.parse_args()

    try:
        df = pd.read_parquet(args.file_path)
        
        print(f"Schema for {args.file_path}:")
        print(df.info())
        
        print(f"\nSample data ({args.sample_size} rows):")
        print(df.head(args.sample_size))

    except Exception as e:
        print(f"Error inspecting file: {e}")

if __name__ == "__main__":
    main()
