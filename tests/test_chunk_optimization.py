#!/usr/bin/env python3
"""
Test optimal chunking parameters for Yiddish text
Evaluates different chunk sizes and overlaps for quality and retrieval performance
"""
import os
import time
from typing import List, Dict, Tuple, Any
import statistics

from core.preprocess import _split_by_chars, load_book_metadata, process_txt_file, _clean_page_text
from core.translate import Translator
from core.embeddings import embed_texts

# Test configurations
CHUNK_CONFIGS = [
    # (size, overlap, description)
    (1500, 150, "Small chunks, low overlap"),
    (2000, 200, "Medium-small chunks, medium overlap"), 
    (2500, 200, "Current setting"),
    (2500, 300, "Current size, higher overlap"),
    (3000, 200, "Larger chunks, medium overlap"),
    (3000, 400, "Larger chunks, very high overlap"),
    (3500, 200, "Very large chunks, medium overlap"),
    (3500, 300, "Very large chunks, high overlap"),
    (3500, 400, "Very large chunks, very high overlap"),
    (4000, 200, "Very very large chunks, medium overlap"),
    (4000, 300, "Very very large chunks, high overlap"),
    (4000, 400, "Very very large chunks, very high overlap"),
]

def get_sample_texts() -> List[str]:
    """Get representative text samples from ALL books with much larger samples"""
    samples = []
    txt_files = [f for f in os.listdir("data/raw") if f.endswith(".txt")]
    
    total_target_chars = 1_000_000  # 200k chars - much more representative
    chars_per_book = total_target_chars // len(txt_files)
    
    print(f"Sampling from {len(txt_files)} books, ~{chars_per_book:,} chars each")
    
    for filename in txt_files:  # Test on ALL books
        with open(f"data/raw/{filename}", "r", encoding="utf-8") as f:
            content = f.read()
            
            if len(content) < chars_per_book:
                # Use entire small book
                sample = content
            else:
                # Take multiple sections from larger books to avoid bias
                section_size = chars_per_book // 3
                sections = []
                
                # Beginning (skip first 500 chars for headers)
                start1 = min(500, len(content) // 10)
                sections.append(content[start1:start1 + section_size])
                
                # Middle  
                start2 = len(content) // 2
                sections.append(content[start2:start2 + section_size])
                
                # End (but not the very end to avoid footers)
                start3 = len(content) - section_size - min(500, len(content) // 10)
                sections.append(content[start3:start3 + section_size])
                
                sample = "\n\n".join(sections)
            
            samples.append(sample)
            print(f"Sample from {filename}: {len(sample):,} chars")
    
    return samples

def analyze_chunks(chunks: List[str], config_name: str) -> Dict[str, Any]:
    """Analyze chunk quality metrics"""
    if not chunks:
        return {"error": "No chunks generated"}
    
    lengths = [len(c) for c in chunks]
    word_counts = [len(c.split()) for c in chunks]
    
    # Look for sentence boundaries - focus on Hebrew/Yiddish text patterns
    sentence_endings = 0
    sentence_breaks = 0
    hebrew_chunks = 0
    
    for chunk in chunks:
        chunk_clean = chunk.strip()
        if not chunk_clean:
            continue
            
        # Check if chunk contains significant Hebrew/Yiddish content
        hebrew_chars = sum(1 for c in chunk_clean if '\u0590' <= c <= '\u05FF')  # Hebrew Unicode block
        hebrew_ratio = hebrew_chars / len(chunk_clean)
        
        if hebrew_ratio > 0.3:  # At least 30% Hebrew characters
            hebrew_chunks += 1
            
            # Count chunks ending with sentence punctuation (including Hebrew punctuation)
            if chunk_clean.endswith(('.', '!', '?', ':', ';', '״', '׳')):
                sentence_endings += 1
            
            # Count internal sentence breaks
            sentence_breaks += chunk_clean.count('.') + chunk_clean.count('!') + chunk_clean.count('?')
        
    # Only calculate percentages for Hebrew-containing chunks
    total_relevant_chunks = hebrew_chunks if hebrew_chunks > 0 else len(chunks)
    
    # Detect potential mid-word breaks (chunks starting with lowercase after space)
    mid_word_breaks = 0
    for chunk in chunks[1:]:  # Skip first chunk
        if chunk and chunk[0].islower() and not chunk.startswith(' '):
            mid_word_breaks += 1
    
    return {
        "config": config_name,
        "total_chunks": len(chunks),
        "hebrew_chunks": hebrew_chunks,
        "hebrew_chunks_pct": hebrew_chunks / len(chunks) * 100 if chunks else 0,
        "avg_length": statistics.mean(lengths),
        "length_std": statistics.stdev(lengths) if len(lengths) > 1 else 0,
        "min_length": min(lengths),
        "max_length": max(lengths),
        "avg_words": statistics.mean(word_counts),
        "sentence_endings_pct": sentence_endings / total_relevant_chunks * 100 if total_relevant_chunks > 0 else 0,
        "avg_sentences_per_chunk": sentence_breaks / total_relevant_chunks if total_relevant_chunks > 0 else 0,
        "mid_word_breaks": mid_word_breaks,
        "mid_word_breaks_pct": mid_word_breaks / len(chunks) * 100,
    }

def test_retrieval_overlap(chunks: List[str], config_name: str) -> Dict[str, float]:
    """Test how much content overlaps between adjacent chunks"""
    if len(chunks) < 2:
        return {"avg_overlap_ratio": 0.0}
    
    overlap_ratios = []
    for i in range(len(chunks) - 1):
        chunk1 = chunks[i]
        chunk2 = chunks[i + 1]
        
        # Find longest common substring at end of chunk1 and start of chunk2
        overlap_len = 0
        min_len = min(len(chunk1), len(chunk2))
        
        for j in range(1, min_len + 1):
            if chunk1[-j:] == chunk2[:j]:
                overlap_len = j
        
        if len(chunk1) > 0:
            overlap_ratio = overlap_len / len(chunk1)
            overlap_ratios.append(overlap_ratio)
    
    return {
        "avg_overlap_ratio": statistics.mean(overlap_ratios) if overlap_ratios else 0.0,
        "max_overlap_ratio": max(overlap_ratios) if overlap_ratios else 0.0,
        "min_overlap_ratio": min(overlap_ratios) if overlap_ratios else 0.0,
    }

def estimate_embedding_costs(chunks: List[str], config_name: str) -> Dict[str, Any]:
    """Estimate embedding costs for different chunk configurations"""
    total_chars = sum(len(c) for c in chunks)
    
    # Use actual tokenization ratios from previous test
    TOKENIZATION_RATIOS = {
        'gemini': 1.97,   # chars per token
        'cohere': 1.78,   # chars per token  
    }
    
    gemini_tokens = total_chars / TOKENIZATION_RATIOS['gemini']
    cohere_tokens = total_chars / TOKENIZATION_RATIOS['cohere']
    
    # Cost per 1M tokens
    EMBEDDING_COST = 0.15  # $0.15/1M tokens
    
    gemini_cost = gemini_tokens * EMBEDDING_COST / 1_000_000
    cohere_cost = cohere_tokens * EMBEDDING_COST / 1_000_000
    total_cost = gemini_cost + cohere_cost
    
    return {
        "total_chars": total_chars,
        "gemini_tokens": int(gemini_tokens),
        "cohere_tokens": int(cohere_tokens),
        "embedding_cost_usd": total_cost,
        "cost_per_chunk": total_cost / len(chunks) if chunks else 0,
    }

def test_chunk_configurations():
    """Test all chunk configurations and compare results"""
    print("=== CHUNK CONFIGURATION OPTIMIZATION TEST ===\n")
    
    # Get sample texts
    sample_texts = get_sample_texts()
    combined_text = "\n\n".join(sample_texts)
    print(f"Combined test text: {len(combined_text):,} characters\n")
    
    results = []
    
    # Test each configuration
    for chunk_size, overlap, description in CHUNK_CONFIGS:
        print(f"Testing: {description} ({chunk_size} chars, {overlap} overlap)")
        
        # Generate chunks
        chunks = _split_by_chars(_clean_page_text(combined_text), chunk_size, overlap)
        
        # Analyze chunk quality
        quality = analyze_chunks(chunks, description)
        overlap_analysis = test_retrieval_overlap(chunks, description)
        cost_analysis = estimate_embedding_costs(chunks, description)
        
        # Combine results
        result = {
            **quality,
            **overlap_analysis,
            **cost_analysis,
            "chunk_size_target": chunk_size,
            "overlap_target": overlap,
        }
        results.append(result)
        
        # Print summary
        print(f"  Chunks: {quality['total_chunks']}")
        print(f"  Avg length: {quality['avg_length']:.0f} chars ({quality['avg_words']:.0f} words)")
        print(f"  Length variation: ±{quality['length_std']:.0f}")
        print(f"  Sentence endings: {quality['sentence_endings_pct']:.1f}%")
        print(f"  Mid-word breaks: {quality['mid_word_breaks_pct']:.1f}%")
        print(f"  Actual overlap: {overlap_analysis['avg_overlap_ratio']*100:.1f}%")
        print(f"  Embedding cost: ${cost_analysis['embedding_cost_usd']:.4f}")
        print()
    
    # Find optimal configuration
    print("=== CONFIGURATION COMPARISON ===")
    print(f"{'Config':<30} {'Chunks':<7} {'Hebrew%':<8} {'AvgLen':<7} {'SentEnd%':<8} {'WordBreak%':<10} {'Overlap%':<8} {'Cost$':<8}")
    print("-" * 100)
    
    for r in results:
        print(f"{r['config']:<30} {r['total_chunks']:<7} {r['hebrew_chunks_pct']:<8.1f} {r['avg_length']:<7.0f} "
              f"{r['sentence_endings_pct']:<8.1f} {r['mid_word_breaks_pct']:<10.1f} "
              f"{r['avg_overlap_ratio']*100:<8.1f} {r['embedding_cost_usd']:<8.4f}")
    
    # Recommendations
    print("\n=== RECOMMENDATIONS ===")
    
    # Best for sentence integrity (highest sentence endings, lowest mid-word breaks)
    sentence_scores = [(r['sentence_endings_pct'] - r['mid_word_breaks_pct'], r) for r in results]
    best_sentence = max(sentence_scores, key=lambda x: x[0])[1]
    print(f"Best sentence integrity: {best_sentence['config']}")
    
    # Best cost efficiency (lowest cost per chunk)
    best_cost = min(results, key=lambda x: x['cost_per_chunk'])
    print(f"Most cost efficient: {best_cost['config']} (${best_cost['cost_per_chunk']:.6f}/chunk)")
    
    # Best balance (good sentence integrity + reasonable cost)
    balance_scores = []
    for r in results:
        sentence_score = r['sentence_endings_pct'] - r['mid_word_breaks_pct']
        cost_score = 1.0 / (r['cost_per_chunk'] * 1000000)  # Invert and scale
        balance = sentence_score * 0.7 + cost_score * 0.3  # Weight sentence quality more
        balance_scores.append((balance, r))
    
    best_balance = max(balance_scores, key=lambda x: x[0])[1]
    print(f"Best overall balance: {best_balance['config']}")
    
    print(f"\nCurrent setting is: {[r for r in results if 'Current setting' in r['config']][0]['config']}")
    
    # Specific recommendations
    print("\n=== SPECIFIC INSIGHTS ===")
    current = next(r for r in results if 'Current setting' in r['config'])
    print(f"Current (2500/200): {current['sentence_endings_pct']:.1f}% sentence endings, "
          f"{current['mid_word_breaks_pct']:.1f}% word breaks, ${current['embedding_cost_usd']:.4f}")
    
    if best_balance['config'] != current['config']:
        print(f"Recommended switch to: {best_balance['config']}")
        print(f"  Benefits: {best_balance['sentence_endings_pct']:.1f}% sentence endings, "
              f"{best_balance['mid_word_breaks_pct']:.1f}% word breaks")
        cost_diff = best_balance['embedding_cost_usd'] - current['embedding_cost_usd']
        print(f"  Cost difference: ${cost_diff:+.4f} ({cost_diff/current['embedding_cost_usd']*100:+.1f}%)")
    else:
        print("Current configuration is already optimal!")
    
    return results

if __name__ == "__main__":
    results = test_chunk_configurations()