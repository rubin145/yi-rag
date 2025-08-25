#!/usr/bin/env python3
"""
New preprocessing script (Step 1).
Converts raw text files to a chunked Parquet format without translation.
"""
import argparse
import logging
import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from core.preprocess import run_preprocess


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess Yiddish corpus: convert raw text files to chunked parquet format (no translation).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic preprocessing
  python scripts/1_preprocess.py

  # Custom input/output paths
  python scripts/1_preprocess.py --input data/raw --output data/processed/chunks_untranslated.parquet
        """
    )
    
    parser.add_argument(
        "--input", "-i",
        default="data/raw",
        help="Input directory containing .txt files (default: data/raw)"
    )
    
    parser.add_argument(
        "--output", "-o", 
        default="data/processed/chunks_untranslated.parquet",
        help="Output parquet file path (default: data/processed/chunks_untranslated.parquet)"
    )
    
    parser.add_argument(
        "--metadata",
        default="data/corpus_metadata.csv",
        help="Path to metadata CSV file (default: data/corpus_metadata.csv)"
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
        help="Remove existing output file before processing."
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

    print(f"🚀 Starting preprocessing (Step 1: No Translation)...")
    print(f"   Input: {args.input}")
    print(f"   Output: {args.output}")
    print(f"   Metadata: {args.metadata}")
    print(f"   Log level: {args.log_level}")
    print()

    try:
        # Temporarily disable translation for this step
        from core.config import settings
        original_translate_setting = settings.translate_chunks_en
        settings.translate_chunks_en = False

        run_preprocess(
            input_dir=args.input,
            output_path=args.output,
            metadata_csv=args.metadata,
            incremental="none"  # Force single file output
        )
        
        # Restore setting
        settings.translate_chunks_en = original_translate_setting

        print("✅ Preprocessing (Step 1) completed successfully!")
        print(f"   📄 Untranslated chunks saved to: {args.output}")
            
    except KeyboardInterrupt:
        print("\n❌ Preprocessing interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        logging.exception("Preprocessing error")
        sys.exit(1)


if __name__ == "__main__":
    main()
