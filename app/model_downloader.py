"""
Model downloader for HF Spaces deployment.
Downloads the sparse model from external hosting.
"""
import requests
import os
from pathlib import Path
from typing import Optional


def download_sparse_model(url: str, local_path: str) -> bool:
    """Download sparse model from URL to local path."""
    try:
        print(f"📥 Downloading sparse model from {url}...")
        response = requests.get(url, stream=True, timeout=60)
        response.raise_for_status()
        
        # Create directory if it doesn't exist
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Write file in chunks
        with open(local_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
        
        print(f"✅ Sparse model downloaded to {local_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download sparse model: {e}")
        return False


def ensure_sparse_model() -> bool:
    """Ensure sparse model exists, download if needed."""
    model_path = "data/sparse_model/tfidf_vectorizer.joblib"
    
    # If model exists locally, we're good
    if os.path.exists(model_path):
        print(f"✅ Sparse model found at {model_path}")
        return True
    
    # Try to download from environment variable or default URL
    model_url = os.getenv("SPARSE_MODEL_URL")
    
    if not model_url:
        print("❌ SPARSE_MODEL_URL not set. Lexical search will be disabled.")
        return False
    
    return download_sparse_model(model_url, model_path)


if __name__ == "__main__":
    # Test the downloader
    ensure_sparse_model()