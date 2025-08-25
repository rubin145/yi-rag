#!/usr/bin/env python3
"""
Calculate costs for the entire corpus using actual Yiddish tokenization ratios
"""
import os
from typing import Dict, Tuple

# Actual tokenization ratios from your test
TOKENIZATION_RATIOS = {
    'tiktoken': 1.24,   # chars per token (for OpenAI, possibly Gemini embeddings)
    'cohere': 1.78,     # chars per token (for Cohere embed-v4.0)
    'gemini': 1.97      # chars per token (for Gemini translation & embeddings)
}

# Cost rates (per million tokens)
COSTS = {
    'translation_input': 0.075,    # Gemini translation input
    'translation_output': 0.30,    # Gemini translation output  
    'embedding': 0.15               # Both Gemini and Cohere embeddings
}

# Chunking settings
CHUNK_SIZE = 2500  # chars per chunk
CHUNK_OVERLAP = 200

def estimate_corpus_size() -> Dict[str, int]:
    """Estimate total corpus size and chunk count"""
    txt_files = [f for f in os.listdir("data/raw") if f.endswith(".txt")]
    
    total_chars = 0
    total_files = 0
    
    for filename in txt_files:
        filepath = f"data/raw/{filename}"
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
            chars = len(content)
            total_chars += chars
            total_files += 1
            print(f"{filename}: {chars:,} chars")
    
    # Estimate chunks (accounting for overlap)
    effective_chunk_size = CHUNK_SIZE - CHUNK_OVERLAP
    estimated_chunks = int(total_chars / effective_chunk_size * 1.1)  # +10% buffer
    
    return {
        'total_files': total_files,
        'total_chars': total_chars,
        'estimated_chunks': estimated_chunks
    }

def calculate_token_costs(chars: int, model: str) -> int:
    """Convert chars to tokens using actual ratios"""
    ratio = TOKENIZATION_RATIOS[model]
    return int(chars / ratio)

def calculate_corpus_costs(corpus_stats: Dict[str, int]) -> Dict[str, float]:
    """Calculate total costs for the corpus"""
    
    total_chars = corpus_stats['total_chars']
    estimated_chunks = corpus_stats['estimated_chunks']
    
    print(f"\n=== CORPUS STATISTICS ===")
    print(f"Files: {corpus_stats['total_files']}")
    print(f"Total characters: {total_chars:,}")
    print(f"Estimated chunks: {estimated_chunks:,}")
    print(f"Chunk size: {CHUNK_SIZE} chars, overlap: {CHUNK_OVERLAP}")
    
    # Translation costs (Yiddish → English)
    yi_tokens_for_translation = calculate_token_costs(total_chars, 'gemini')
    
    # Assuming translation output is ~1.2x input (Yiddish → English expansion)
    en_tokens_from_translation = int(yi_tokens_for_translation * 1.2)
    en_chars_from_translation = en_tokens_from_translation * TOKENIZATION_RATIOS['gemini']
    
    translation_input_cost = yi_tokens_for_translation * COSTS['translation_input'] / 1_000_000
    translation_output_cost = en_tokens_from_translation * COSTS['translation_output'] / 1_000_000
    total_translation_cost = translation_input_cost + translation_output_cost
    
    print(f"\n=== TRANSLATION COSTS (Yiddish → English) ===")
    print(f"Yiddish input: {total_chars:,} chars → {yi_tokens_for_translation:,} tokens")
    print(f"English output: ~{en_tokens_from_translation:,} tokens (1.2x expansion)")
    print(f"Input cost: ${translation_input_cost:.4f}")
    print(f"Output cost: ${translation_output_cost:.4f}")
    print(f"Total translation: ${total_translation_cost:.4f}")
    
    # Embedding costs
    # Yiddish text: 2 models (Gemini + Cohere)
    yi_tokens_gemini = calculate_token_costs(total_chars, 'gemini')
    yi_tokens_cohere = calculate_token_costs(total_chars, 'cohere')
    
    yi_embedding_cost_gemini = yi_tokens_gemini * COSTS['embedding'] / 1_000_000
    yi_embedding_cost_cohere = yi_tokens_cohere * COSTS['embedding'] / 1_000_000
    
    # English text: 1 model (Gemini)
    en_tokens_gemini = calculate_token_costs(en_chars_from_translation, 'gemini')
    en_embedding_cost_gemini = en_tokens_gemini * COSTS['embedding'] / 1_000_000
    
    total_embedding_cost = yi_embedding_cost_gemini + yi_embedding_cost_cohere + en_embedding_cost_gemini
    
    print(f"\n=== EMBEDDING COSTS ===")
    print(f"Yiddish → Gemini: {yi_tokens_gemini:,} tokens → ${yi_embedding_cost_gemini:.4f}")
    print(f"Yiddish → Cohere: {yi_tokens_cohere:,} tokens → ${yi_embedding_cost_cohere:.4f}")
    print(f"English → Gemini: {en_tokens_gemini:,} tokens → ${en_embedding_cost_gemini:.4f}")
    print(f"Total embedding: ${total_embedding_cost:.4f}")
    
    # Grand total
    grand_total = total_translation_cost + total_embedding_cost
    
    print(f"\n=== GRAND TOTAL ===")
    print(f"Translation: ${total_translation_cost:.4f}")
    print(f"Embeddings: ${total_embedding_cost:.4f}")
    print(f"TOTAL COST: ${grand_total:.2f}")
    
    # Per book breakdown
    cost_per_book = grand_total / corpus_stats['total_files']
    print(f"Average per book: ${cost_per_book:.4f}")
    
    return {
        'translation_cost': total_translation_cost,
        'embedding_cost': total_embedding_cost,
        'total_cost': grand_total,
        'cost_per_book': cost_per_book,
        'yi_tokens_gemini': yi_tokens_gemini,
        'yi_tokens_cohere': yi_tokens_cohere,
        'en_tokens_gemini': en_tokens_gemini
    }

def main():
    print("=== YIDDISH CORPUS COST CALCULATOR ===")
    print("Using actual tokenization ratios from test:")
    for model, ratio in TOKENIZATION_RATIOS.items():
        print(f"  {model}: {ratio} chars/token")
    print()
    
    # Get corpus statistics
    corpus_stats = estimate_corpus_size()
    
    # Calculate costs
    costs = calculate_corpus_costs(corpus_stats)
    
    # Show token distribution
    print(f"\n=== TOKEN BREAKDOWN ===")
    total_tokens = costs['yi_tokens_gemini'] + costs['yi_tokens_cohere'] + costs['en_tokens_gemini']
    print(f"Total tokens to process: {total_tokens:,}")
    print(f"  - Yiddish (Gemini): {costs['yi_tokens_gemini']:,}")
    print(f"  - Yiddish (Cohere): {costs['yi_tokens_cohere']:,}")  
    print(f"  - English (Gemini): {costs['en_tokens_gemini']:,}")
    
    return costs

if __name__ == "__main__":
    costs = main()