#!/usr/bin/env python3
"""
Debug chunk endings to see what's actually happening
"""
import os
import random
from core.preprocess import _split_by_chars

def analyze_chunk_endings():
    """Look at actual chunk endings to understand the low sentence ending %"""
    print("=== CHUNK ENDINGS ANALYSIS ===\n")
    
    # Get sample text from different books
    txt_files = [f for f in os.listdir("data/raw") if f.endswith(".txt")]
    
    for book_idx, filename in enumerate(txt_files[:3]):  # Check 3 books
        print(f"\n{'='*60}")
        print(f"BOOK {book_idx+1}: {filename}")
        print(f"{'='*60}")
        
        with open(f"data/raw/{filename}", "r", encoding="utf-8") as f:
            content = f.read()
        
        # Take middle sample to avoid headers/footers
        start = len(content) // 3
        sample = content[start:start + 30000]  # 30k chars
        
        # Test current setting vs recommended
        configs = [
            (2500, 200, "Current (2500/200)"),
            (3000, 200, "Recommended (3000/200)"),
        ]
        
        for chunk_size, overlap, config_name in configs:
            print(f"\n--- {config_name} ---")
            chunks = _split_by_chars(sample, chunk_size, overlap)
            
            # Check Hebrew content
            hebrew_chunks = []
            for chunk in chunks:
                hebrew_chars = sum(1 for c in chunk if '\u0590' <= c <= '\u05FF')
                hebrew_ratio = hebrew_chars / len(chunk) if chunk else 0
                if hebrew_ratio > 0.3:  # At least 30% Hebrew
                    hebrew_chunks.append(chunk)
            
            print(f"Generated {len(chunks)} total chunks, {len(hebrew_chunks)} Hebrew chunks")
            
            # Sample random Hebrew chunks
            sample_chunks = random.sample(hebrew_chunks, min(5, len(hebrew_chunks)))
            
            sentence_ending_chunks = 0
            for i, chunk in enumerate(sample_chunks):
                chunk_clean = chunk.strip()
                last_char = chunk_clean[-1] if chunk_clean else ''
                ends_with_sentence = last_char in '.!?:;״׳'
                
                if ends_with_sentence:
                    sentence_ending_chunks += 1
                
                print(f"\nChunk {i+1} ({len(chunk_clean)} chars):")
                print(f"  Last char: '{last_char}' ({'✓ SENTENCE' if ends_with_sentence else '✗ NO SENTENCE'})")
                print(f"  Last 150 chars:")
                print(f"    ...{chunk_clean[-150:]}")
                
                # Show Hebrew vs non-Hebrew chars in ending
                ending = chunk_clean[-50:]
                hebrew_in_ending = sum(1 for c in ending if '\u0590' <= c <= '\u05FF')
                print(f"  Ending has {hebrew_in_ending}/50 Hebrew chars ({hebrew_in_ending/50*100:.0f}%)")
            
            pct = sentence_ending_chunks / len(sample_chunks) * 100 if sample_chunks else 0
            print(f"\n  Sample sentence endings: {sentence_ending_chunks}/{len(sample_chunks)} ({pct:.1f}%)")

def analyze_punctuation_patterns():
    """Look at what punctuation actually appears in the text"""
    print(f"\n{'='*60}")
    print("PUNCTUATION PATTERN ANALYSIS")
    print(f"{'='*60}")
    
    txt_files = [f for f in os.listdir("data/raw") if f.endswith(".txt")]
    filename = txt_files[0]  # Just one book
    
    with open(f"data/raw/{filename}", "r", encoding="utf-8") as f:
        content = f.read()
    
    # Take middle sample
    start = len(content) // 3
    sample = content[start:start + 50000]  # 50k chars
    
    print(f"Analyzing {filename} (50k char sample)")
    
    # Count all punctuation
    punctuation = {
        '.': 'Period',
        '!': 'Exclamation',
        '?': 'Question', 
        ':': 'Colon',
        ';': 'Semicolon',
        ',': 'Comma',
        '״': 'Hebrew quote end',
        '׳': 'Hebrew geresh',
        '—': 'Em dash',
        '-': 'Hyphen',
        '(': 'Open paren',
        ')': 'Close paren',
        '"': 'Quote',
        "'": 'Apostrophe',
    }
    
    counts = {}
    for punct in punctuation.keys():
        count = sample.count(punct)
        if count > 0:
            counts[punct] = count
    
    print(f"\nPunctuation frequency in {len(sample):,} chars:")
    for punct, count in sorted(counts.items(), key=lambda x: x[1], reverse=True):
        rate_per_1000 = count / len(sample) * 1000
        print(f"  '{punct}' ({punctuation[punct]}): {count:,} times ({rate_per_1000:.1f} per 1000 chars)")
    
    # Show some context around periods
    if '.' in sample:
        print(f"\nContext around periods (first 5 occurrences):")
        period_positions = [i for i, c in enumerate(sample) if c == '.']
        for i, pos in enumerate(period_positions[:5]):
            start_ctx = max(0, pos - 30)
            end_ctx = min(len(sample), pos + 30)
            context = sample[start_ctx:end_ctx]
            context = context.replace('\n', '\\n')
            print(f"  {i+1}: ...{context}...")

if __name__ == "__main__":
    analyze_chunk_endings()
    analyze_punctuation_patterns()