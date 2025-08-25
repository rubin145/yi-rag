#!/usr/bin/env python3
"""
Test actual tokenization ratios for Yiddish text across different models
"""
import os
from typing import List, Dict, Tuple

def get_sample_yiddish_text() -> List[str]:
    """Get sample Yiddish text from corpus files"""
    samples = []
    txt_files = [f for f in os.listdir("data/raw") if f.endswith(".txt")]
    
    for filename in txt_files[:3]:  # First 3 files
        with open(f"data/raw/{filename}", "r", encoding="utf-8") as f:
            content = f.read()
            # Take first 1000 chars as sample
            if len(content) > 1000:
                sample = content[500:1500]  # Skip potential headers
                samples.append(sample.strip())
    
    return samples

def test_tiktoken_tokenization(texts: List[str]) -> Dict[str, float]:
    """Test cl100k_base tokenization (used by OpenAI/possibly Gemini)"""
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        
        total_chars = 0
        total_tokens = 0
        
        for text in texts:
            chars = len(text)
            tokens = len(enc.encode(text))
            total_chars += chars
            total_tokens += tokens
            print(f"  Sample: {chars} chars → {tokens} tokens (ratio: {chars/tokens:.2f})")
        
        avg_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        return {
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "avg_chars_per_token": avg_ratio
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}

def test_cohere_tokenization(texts: List[str]) -> Dict[str, float]:
    """Test Cohere tokenization"""
    try:
        import cohere
        from core.config import settings
        
        if not settings.cohere_api_key:
            return {"error": "COHERE_API_KEY not set"}
            
        client = cohere.Client(api_key=settings.cohere_api_key)
        
        total_chars = 0
        total_tokens = 0
        
        for i, text in enumerate(texts):
            try:
                # Use tokenize endpoint
                resp = client.tokenize(text=text, model="embed-v4.0")
                chars = len(text)
                tokens = len(resp.tokens)
                total_chars += chars
                total_tokens += tokens
                print(f"  Sample {i+1}: {chars} chars → {tokens} tokens (ratio: {chars/tokens:.2f})")
            except Exception as e:
                print(f"  Sample {i+1} failed: {e}")
        
        avg_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        return {
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "avg_chars_per_token": avg_ratio
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}

def test_gemini_tokenization(texts: List[str]) -> Dict[str, float]:
    """Test Gemini tokenization"""
    try:
        import google.generativeai as genai
        from core.config import settings
        
        if not settings.gemini_api_key:
            return {"error": "GEMINI_API_KEY not set"}
            
        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel('gemini-1.5-flash')
        
        total_chars = 0
        total_tokens = 0
        
        for i, text in enumerate(texts):
            try:
                # Count tokens using Gemini's count_tokens method
                resp = model.count_tokens(text)
                chars = len(text)
                tokens = resp.total_tokens
                total_chars += chars
                total_tokens += tokens
                print(f"  Sample {i+1}: {chars} chars → {tokens} tokens (ratio: {chars/tokens:.2f})")
            except Exception as e:
                print(f"  Sample {i+1} failed: {e}")
        
        avg_ratio = total_chars / total_tokens if total_tokens > 0 else 0
        return {
            "total_chars": total_chars,
            "total_tokens": total_tokens,
            "avg_chars_per_token": avg_ratio
        }
    except Exception as e:
        print(f"  Error: {e}")
        return {"error": str(e)}

def main():
    print("=== YIDDISH TEXT TOKENIZATION TEST ===\n")
    
    # Get sample texts
    samples = get_sample_yiddish_text()
    if not samples:
        print("No sample texts found!")
        return
    
    print(f"Testing with {len(samples)} sample texts:")
    for i, sample in enumerate(samples):
        print(f"  Sample {i+1}: {len(sample)} chars")
        print(f"    Preview: {sample[:60]}...")
    print()
    
    # Test each tokenization method
    results = {}
    
    print("1. Testing tiktoken (cl100k_base - used by OpenAI, possibly Gemini embeddings):")
    results['tiktoken'] = test_tiktoken_tokenization(samples)
    print()
    
    print("2. Testing Cohere tokenization:")
    results['cohere'] = test_cohere_tokenization(samples)
    print()
    
    print("3. Testing Gemini tokenization:")
    results['gemini'] = test_gemini_tokenization(samples)
    print()
    
    # Summary
    print("=== SUMMARY ===")
    for model_name, result in results.items():
        if 'error' in result:
            print(f"{model_name.upper()}: Error - {result['error']}")
        else:
            ratio = result.get('avg_chars_per_token', 0)
            print(f"{model_name.upper()}: {ratio:.2f} chars per token")
    
    print("\nUse these ratios for accurate cost estimation!")
    
    return results

if __name__ == "__main__":
    main()