#!/usr/bin/env python3
"""
Test the full preprocessing + indexing pipeline with a small sample
"""
import os
import time
from typing import List, Dict, Any

from core.config import settings
from core.preprocess import process_txt_file, load_book_metadata
from core.translate import Translator
from core.embeddings import embed_texts
from core.store_qdrant import ensure_collection_named_vectors, upsert_batch_with_named_vectors
from scripts.index_corpus import load_vector_specs

def test_preprocessing():
    """Test preprocessing with one book, limited chunks"""
    print("=== TESTING PREPROCESSING ===")
    
    # Find smallest book for testing
    txt_files = [(f, os.path.getsize(f"data/raw/{f}")) for f in os.listdir("data/raw") if f.endswith(".txt")]
    txt_files.sort(key=lambda x: x[1])  # Sort by file size
    test_file = txt_files[0][0]  # Smallest file
    
    print(f"Using smallest book: {test_file} ({txt_files[0][1]:,} bytes)")
    
    # Load metadata and process
    metadata = load_book_metadata("data/corpus_metadata.csv")
    translator = Translator()
    
    chunks, stats, book_id = process_txt_file(f"data/raw/{test_file}", metadata, translator)
    
    print(f"Book ID: {book_id}")
    print(f"Total chunks generated: {len(chunks)}")
    print(f"Processing stats: {stats}")
    
    # Show sample chunk with all metadata
    if chunks:
        chunk = chunks[0]
        print(f"\n=== SAMPLE CHUNK ===")
        print(f"ID: {chunk.id}")
        print(f"Book: {chunk.title_en} ({chunk.title_yi})")
        print(f"Author: {chunk.author} ({chunk.author_yi})")
        print(f"Year: {chunk.year}, Place: {chunk.place}")
        print(f"Publisher: {chunk.publisher}")
        print(f"Storage: {chunk.storage_locations}")
        print(f"Subjects: {chunk.subjects}")
        print(f"Total pages: {chunk.total_pages}")
        print(f"URLs: {chunk.source_url}")
        print(f"Yi text ({len(chunk.yi_text)} chars): {chunk.yi_text[:100]}...")
        if chunk.tr_en_text:
            print(f"En text ({len(chunk.tr_en_text)} chars): {chunk.tr_en_text[:100]}...")
            print(f"Translation metadata: {chunk.tr_en_metadata}")
    
    # Limit to first few chunks for testing
    test_chunks = chunks[:3]
    print(f"\nUsing first {len(test_chunks)} chunks for embedding test")
    
    return test_chunks, book_id

def test_embeddings(chunks: List) -> Dict[str, List]:
    """Test embeddings with sample chunks"""
    print(f"\n=== TESTING EMBEDDINGS ===")
    
    # Load vector specs
    vector_specs = load_vector_specs("config/vectors.yaml", None)
    print(f"Vector specs loaded: {[s.name for s in vector_specs]}")
    
    # Convert chunks to rows (like pandas would)
    rows = [chunk.model_dump() for chunk in chunks]
    
    vectors_by_name = {}
    
    # Test each vector spec
    for spec in vector_specs:
        print(f"\nTesting {spec.name} ({spec.provider}/{spec.model})...")
        
        # Extract source texts
        texts = [row.get(spec.source_field) for row in rows]
        print(f"  Source texts: {len([t for t in texts if t])} non-empty")
        
        # Generate embeddings
        t0 = time.time()
        vectors = embed_texts(texts, spec.__dict__)
        dt = time.time() - t0
        
        success_count = sum(1 for v in vectors if v is not None)
        print(f"  Generated: {success_count}/{len(vectors)} vectors in {dt:.2f}s")
        
        if success_count > 0:
            # Show sample vector info
            sample_vec = next(v for v in vectors if v is not None)
            print(f"  Sample vector: dim={len(sample_vec)}, first 5 values: {sample_vec[:5]}")
        
        vectors_by_name[spec.name] = vectors
    
    return vectors_by_name

def test_qdrant_indexing(chunks: List, vectors_by_name: Dict[str, List]):
    """Test Qdrant collection creation and indexing"""
    print(f"\n=== TESTING QDRANT INDEXING ===")
    
    # Load vector specs for collection setup
    vector_specs = load_vector_specs("config/vectors.yaml", None)
    
    # Create test collection name
    test_collection = f"{settings.collection_name}_test"
    print(f"Creating test collection: {test_collection}")
    
    # Ensure collection exists (recreate for clean test)
    ensure_collection_named_vectors(
        test_collection, 
        [s.__dict__ for s in vector_specs], 
        recreate=True
    )
    
    # Convert chunks to rows
    rows = [chunk.model_dump() for chunk in chunks]
    
    # Upsert vectors
    print(f"Upserting {len(rows)} chunks with vectors...")
    upsert_batch_with_named_vectors(test_collection, rows, vectors_by_name)
    
    # Verify upload
    from core.store_qdrant import get_client
    client = get_client()
    
    try:
        collection_info = client.get_collection(test_collection)
        print(f"Collection created successfully!")
        print(f"  Vectors count: {collection_info.vectors_count}")
        print(f"  Points count: {collection_info.points_count}")
        
        # Sample a point to verify payload
        points = client.scroll(collection_name=test_collection, limit=1, with_payload=True, with_vectors=False)
        if points[0]:
            sample_point = points[0][0]
            print(f"\n=== SAMPLE INDEXED POINT ===")
            print(f"ID: {sample_point.id}")
            print(f"Payload keys: {list(sample_point.payload.keys())}")
            print(f"Book: {sample_point.payload.get('title_en')} ({sample_point.payload.get('year')})")
            print(f"Author: {sample_point.payload.get('author')}")
            print(f"Metadata fields: place={sample_point.payload.get('place')}, publisher={sample_point.payload.get('publisher')}")
    
    except Exception as e:
        print(f"Error verifying collection: {e}")
    
    print(f"\nTest collection '{test_collection}' ready for cleanup or further testing!")

def main():
    """Run full pipeline test"""
    print("=== FULL PIPELINE TEST ===")
    print(f"Settings: chunk_strategy={settings.chunk_strategy}, translate={settings.translate_chunks_en}")
    print()
    
    try:
        # Step 1: Test preprocessing
        chunks, book_id = test_preprocessing()
        
        # Step 2: Test embeddings
        vectors_by_name = test_embeddings(chunks)
        
        # Step 3: Test Qdrant indexing
        test_qdrant_indexing(chunks, vectors_by_name)
        
        print("\n🎉 FULL PIPELINE TEST COMPLETED SUCCESSFULLY!")
        print("Ready to run on full corpus:")
        print("  1. python core/preprocess.py")
        print("  2. python scripts/index_corpus.py --recreate")
        
    except Exception as e:
        print(f"\n❌ PIPELINE TEST FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()