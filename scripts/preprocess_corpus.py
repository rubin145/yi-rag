#!/usr/bin/env python3
"""
Preprocess corpus script - converts raw text files to chunked parquet format.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from core.preprocess import run_preprocess


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Yiddish corpus: convert raw text files to chunked parquet format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing (single parquet file)
  python scripts/preprocess_corpus.py

  # Process with append mode (incremental processing)
  python scripts/preprocess_corpus.py --mode append

  # Process by book (separate parquet per book)
  python scripts/preprocess_corpus.py --mode book

  # Custom input/output paths
  python scripts/preprocess_corpus.py --input data/raw --output data/processed/my_chunks.parquet

  # With debug logging
  python scripts/preprocess_corpus.py --log-level DEBUG
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Input directory containing .txt files (default: data/raw)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="data/processed/chunks.parquet",
        help="Output parquet file path (default: data/processed/chunks.parquet)"
    )
    
    parser.add_argument(
        "--metadata",
        default="data/corpus_metadata.csv",
        help="Path to metadata CSV file (default: data/corpus_metadata.csv)"
    )
    
    parser.add_argument(
        "--mode",
        choices=["none", "book", "append"],
        default="none",
        help="""Processing mode:
        - none: Single parquet file (default)
        - book: Separate parquet per book in data/processed/parts/
        - append: Append to existing parquet file"""
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Remove existing output file before processing (useful for append mode schema issues)"
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s"
    )

    # Validate inputs
    if not os.path.exists(args.input):
        print(f"Error: Input directory '{args.input}' does not exist")
        sys.exit(1)

    if args.metadata and not os.path.exists(args.metadata):
        print(f"Warning: Metadata file '{args.metadata}' not found, continuing without metadata")

    # Clean existing output file if requested
    if args.clean and os.path.exists(args.output):
        print(f"🧹 Removing existing output file: {args.output}")
        os.remove(args.output)

    print(f"🚀 Starting preprocessing...")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Mode: {args.mode}")
    print(f"   Metadata: {args.metadata}")
    print(f"   Log level: {args.log_level}")
    print()

    try:
        run_preprocess(
            input_dir=args.input,
            output_path=args.output,
            metadata_csv=args.metadata,
            incremental=args.mode
        )
        print("✅ Preprocessing completed successfully!")
        
        if args.mode == "book":
            print(f"   📁 Individual book files saved to: data/processed/parts/")
        else:
            print(f"   📄 Chunks saved to: {args.output}")
            
    except KeyboardInterrupt:
        print("\n❌ Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        logging.exception("Preprocessing error")
        sys.exit(1)


if __name__ == "__main__":
    main()

#poetry run python scripts/preprocess_corpus.py --mode append --log-level DEBUG