#!/usr/bin/env python3
"""
Test intelligent chunking for both start and end boundaries,
including improved overlap quality check
"""
import os
import string
from core.preprocess import _split_by_chars, _clean_page_text


def analyze_chunk_boundaries():
    """Analyze both start and end boundaries of chunks"""
    print("=== INTELLIGENT CHUNKING BOUNDARY ANALYSIS ===\n")
    
    txt_files = [f for f in os.listdir("data/raw") if f.endswith(".txt")]
    filename = txt_files[2]  # Test one book
    
    with open(f"data/raw/{filename}", "r", encoding="utf-8") as f:
        content = f.read()
    
    
    # Take start samples
    start = 0
    sample = content[start:start + 30000]  # 30k chars

    # Take middle sample
    #start = len(content) // 3
    #sample = content[start:start + 30000]  # 30k chars
    
    # Take end samples
    #sample = content[-30000:]  # últimos 30k chars
    #start = len(content) - len(sample)


    sample = _clean_page_text(sample)

    print(f"Testing: {filename}")
    print(f"Sample: {len(sample):,} characters\n")
    
    # Test with current recommendation
    chunks = _split_by_chars(sample, 3000, 200)
    print(f"Generated {len(chunks)} chunks\n")
    
    # Analyze first few chunks
    good_starts = 0
    good_ends = 0
    
    for i, chunk in enumerate(chunks[:8]):  # First 8 chunks
        chunk_clean = chunk.strip()
        
        # Analyze start
        first_char = chunk_clean[0] if chunk_clean else ''
        first_50 = chunk_clean[:50] if len(chunk_clean) >= 50 else chunk_clean
        
        # Good start indicators
        starts_with_capital = first_char.isupper() if first_char.isalpha() else False
        
        good_start = (
            starts_with_capital or 
            first_char in 'אבגדהוזחטיכלמנסעפצקרשת' or  # Hebrew letters
            first_50.strip().startswith('—') or  # Dialog
            first_50.strip().startswith('"') or  # Quote
            i == 0  # First chunk always good
        )
        
        if good_start:
            good_starts += 1
        
        # Analyze end
        last_char = chunk_clean[-1] if chunk_clean else ''
        last_50 = chunk_clean[-50:] if len(chunk_clean) >= 50 else chunk_clean
        
        good_end = last_char in '.!?:;״׳'
        if good_end:
            good_ends += 1
        
        print(f"Chunk {i+1} ({len(chunk_clean)} chars):")
        print(f"  START: '{first_char}' ({'✓' if good_start else '✗'}) - {first_50}...")
        print(f"  END:   '{last_char}' ({'✓' if good_end else '✗'}) - ...{last_50}")
        print()
    
    # Summary
    start_pct = good_starts / min(8, len(chunks)) * 100
    end_pct = good_ends / min(8, len(chunks)) * 100
    
    print("=== SUMMARY ===")
    print(f"Good starts: {good_starts}/{min(8, len(chunks))} ({start_pct:.1f}%)")
    print(f"Good ends: {good_ends}/{min(8, len(chunks))} ({end_pct:.1f}%)")
    
    if start_pct >= 75 and end_pct >= 75:
        print("✅ Intelligent chunking working well!")
    elif end_pct >= 75:
        print("⚠️  Endings good, but starts need improvement")
    elif start_pct >= 75:
        print("⚠️  Starts good, but endings need improvement") 
    else:
        print("❌ Both starts and endings need improvement")


# === NUEVO CÓDIGO PARA OVERLAP ===
def _is_boundary(text: str, pos: int) -> bool:
    """Check if pos is a safe boundary (start or end of overlap)."""
    if pos <= 0 or pos >= len(text):
        return True
    return text[pos-1] in string.whitespace + string.punctuation


def compare_overlap_quality():
    """Check if overlap areas make sense with true boundary detection"""
    print(f"\n{'='*60}")
    print("OVERLAP QUALITY ANALYSIS") 
    print(f"{'='*60}")
    
    txt_files = [f for f in os.listdir("data/raw") if f.endswith(".txt")]
    filename = txt_files[0]
    
    with open(f"data/raw/{filename}", "r", encoding="utf-8") as f:
        content = f.read()
    
    sample = content[len(content)//3:len(content)//3 + 20000]
    chunks = _split_by_chars(sample, 3000, 200)
    
    print(f"Analyzing overlap between chunks...\n")
    
    for i in range(min(3, len(chunks) - 1)):
        chunk1 = chunks[i].strip()
        chunk2 = chunks[i + 1].strip()
        
        # Find actual overlap
        max_overlap = min(len(chunk1), len(chunk2))
        actual_overlap = 0
        for j in range(1, max_overlap + 1):
            if chunk1[-j:] == chunk2[:j]:
                actual_overlap = j
        
        print(f"Chunks {i+1} → {i+2}:")
        print(f"  Overlap: {actual_overlap} chars")
        
        if actual_overlap > 0:
            overlap_text = chunk1[-actual_overlap:]
            print(f"  Overlap text: '{overlap_text[:80]}...'")
            
            # Context check: char before overlap in chunk1 and char after overlap in chunk2
            cuts_word = False
            if len(chunk1) > actual_overlap:
                before = chunk1[-(actual_overlap+1)]
                if before.isalnum() and overlap_text[0].isalnum():
                    cuts_word = True
            if len(chunk2) > actual_overlap:
                after = chunk2[actual_overlap]
                if after.isalnum() and overlap_text[-1].isalnum():
                    cuts_word = True
            
            if cuts_word:
                print("  ⚠️ Overlap cuts through a word")
            else:
                print("  ✅ Overlap aligned cleanly")
        else:
            print("  ❌ No overlap detected")
        print()



if __name__ == "__main__":
    analyze_chunk_boundaries()
    compare_overlap_quality()
