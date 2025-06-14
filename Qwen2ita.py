#!/usr/bin/env python3
"""
OCR Text Processing Script with Proper Chunking for Qwen2-1.5B-Ita
Modified to process only keys "1" to "6"
"""

import json
import re
from pathlib import Path
from pprint import pprint
import sys

def clean_ocr_text(text: str) -> str:
    """Preprocess OCR text by cleaning common OCR artifacts"""
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"-\s+", "", text)
    text = re.sub(r"\s+([.,!?;:])", r"\1", text)
    return text.strip()

def estimate_tokens(text: str) -> int:
    """Rough estimation of tokens (1 token ≈ 4 characters for most languages)"""
    return len(text) // 4

def filter_by_keys_and_tokens(data_dict: dict, target_keys: list, max_tokens: int = None) -> dict:
    """Filter dataset to include only specific keys and optionally limit tokens"""
    # First filter by keys
    filtered_data = {k: v for k, v in data_dict.items() if k in target_keys}
    
    print(f"🎯 Filtering dataset to keys: {target_keys}")
    print(f"✅ Found {len(filtered_data)}/{len(target_keys)} target keys in dataset")
    
    # Show which keys were found and which were missing
    found_keys = set(filtered_data.keys())
    missing_keys = set(target_keys) - found_keys
    if missing_keys:
        print(f"⚠️ Missing keys: {sorted(missing_keys)}")
    if found_keys:
        print(f"✅ Found keys: {sorted(found_keys)}")
    
    # Apply token filtering if specified
    if max_tokens:
        print(f"🎯 Additional filtering to max {max_tokens:,} tokens...")
        final_data = {}
        total_tokens = 0
        
        # Process keys in order 1-6
        for key in sorted(filtered_data.keys(), key=lambda x: int(x)):
            text = filtered_data[key]
            text_tokens = estimate_tokens(text)
            
            if total_tokens + text_tokens <= max_tokens:
                final_data[key] = text
                total_tokens += text_tokens
            else:
                # If adding this sample would exceed limit, check if we can fit a truncated version
                remaining_tokens = max_tokens - total_tokens
                if remaining_tokens > 100:  # Only if we have at least 100 tokens left
                    # Truncate the text to fit remaining tokens
                    chars_to_keep = remaining_tokens * 4
                    truncated_text = text[:chars_to_keep]
                    final_data[key] = truncated_text
                    total_tokens = max_tokens
                break
        
        print(f"✅ After token filtering: {len(final_data)} samples (~{total_tokens:,} tokens)")
        return final_data
    
    return filtered_data

def load_data(base_path: Path, target_keys: list, max_tokens: int = None):
    """Load cleaned and OCR data from JSON files with key and token filtering"""
    cleaned_path = base_path / "cleaned.json"
    ocr_path = base_path / "original_ocr.json"
    
    print(f"📂 Loading data from {base_path}")
    
    if not cleaned_path.exists():
        raise FileNotFoundError(f"Cleaned data file not found: {cleaned_path}")
    if not ocr_path.exists():
        raise FileNotFoundError(f"OCR data file not found: {ocr_path}")
    
    with open(cleaned_path, encoding="utf-8") as f:
        cleaned_data = json.load(f)
    
    with open(ocr_path, encoding="utf-8") as f:
        ocr_data = json.load(f)
    
    print(f"✅ Loaded {len(cleaned_data)} cleaned samples")
    print(f"✅ Loaded {len(ocr_data)} OCR samples")
    
    # Apply key and token filtering
    ocr_data = filter_by_keys_and_tokens(ocr_data, target_keys, max_tokens)
    # Filter cleaned_data to match the same keys
    cleaned_data = {k: v for k, v in cleaned_data.items() if k in ocr_data}
    
    print(f"📊 Final dataset: {len(ocr_data)} samples")
    
    return cleaned_data, ocr_data

def split_text_into_chunks(text: str, max_chars: int = 800) -> list:
    """
    Split long text into smaller chunks for processing.
    Tries to split at sentence boundaries to maintain context.
    """
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    
    # First try to split by sentences (periods, exclamation marks, question marks)
    sentences = re.split(r'([.!?]+)', text)
    
    # Reconstruct sentences with their punctuation
    reconstructed_sentences = []
    for i in range(0, len(sentences) - 1, 2):
        sentence = sentences[i].strip()
        if i + 1 < len(sentences):
            punctuation = sentences[i + 1]
            sentence += punctuation
        if sentence:
            reconstructed_sentences.append(sentence)
    
    # If no sentences found, fall back to splitting by commas or spaces
    if not reconstructed_sentences:
        if ',' in text:
            reconstructed_sentences = [s.strip() for s in text.split(',') if s.strip()]
        else:
            # Last resort: split by words
            words = text.split()
            chunk_size = max_chars // 10  # Rough estimate of words per chunk
            reconstructed_sentences = [' '.join(words[i:i+chunk_size]) 
                                     for i in range(0, len(words), chunk_size)]
    
    # Group sentences into chunks
    current_chunk = ""
    for sentence in reconstructed_sentences:
        # If adding this sentence would exceed the limit and we have content, save chunk
        if len(current_chunk) + len(sentence) + 1 > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    # Add the last chunk if it has content
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

def try_import_transformers():
    """Try to import transformers with better error handling"""
    try:
        print("🔍 Checking transformers installation...")
        
        # Try basic import first
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        # Check if version is sufficient
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 4 or (major == 4 and minor < 37):
            print(f"⚠️ Warning: Transformers version {transformers.__version__} may not support Qwen2")
            print("   Recommended: >= 4.37.0")
        
        return transformers
        
    except ImportError as e:
        print(f"❌ Cannot import transformers: {e}")
        print("\n🔧 Please install transformers:")
        print("   pip install transformers>=4.37.0")
        return None
    except Exception as e:
        print(f"❌ Error with transformers: {e}")
        return None

def initialize_model_safe():
    """Initialize model with safe error handling"""
    transformers = try_import_transformers()
    if not transformers:
        return None, None
    
    print("\n🤖 Initializing Qwen2-1.5B-Ita model...")
    print("💾 Note: Model will be downloaded only once (cached), but loaded into memory each run")
    
    try:
        from transformers import pipeline
        print("📝 Loading with pipeline...")
        
        # Try with trust_remote_code=True
        pipe = pipeline(
            "text-generation", 
            model="DeepMount00/Qwen2-1.5B-Ita",
            trust_remote_code=True,
            device_map="auto" if torch_available() else None
        )
        
        print("✅ Model loaded successfully with pipeline")
        return pipe, "pipeline"
        
    except Exception as e:
        print(f"⚠️ Pipeline method failed: {e}")
        
        # Try direct loading
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM
            print("📝 Trying direct model loading...")
            
            tokenizer = AutoTokenizer.from_pretrained(
                "DeepMount00/Qwen2-1.5B-Ita", 
                trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                "DeepMount00/Qwen2-1.5B-Ita", 
                trust_remote_code=True
            )
            
            print("✅ Model loaded successfully with direct method")
            return (model, tokenizer), "direct"
            
        except Exception as e2:
            print(f"❌ Direct loading also failed: {e2}")
            
            print("\n🔧 Troubleshooting suggestions:")
            print("1. Create a virtual environment:")
            print("   python -m venv qwen2_env")
            print("   source qwen2_env/bin/activate  # On Windows: qwen2_env\\Scripts\\activate")
            print("2. Install packages:")
            print("   pip install torch transformers>=4.37.0 accelerate")
            print("3. Try running the script again")
            
            return None, None

def torch_available():
    """Check if torch is available"""
    try:
        import torch
        return True
    except ImportError:
        return False

def generate_text_safe(model_info, prompt, max_new_tokens=200):
    """Generate text with safe error handling"""
    if not model_info[0]:
        return f"[MODEL NOT AVAILABLE] Prompt was: {prompt[:100]}..."
    
    model_obj, model_type = model_info
    
    try:
        if model_type == "pipeline":
            result = model_obj(prompt, max_new_tokens=max_new_tokens, do_sample=False)
            generated = result[0]["generated_text"]
            
            # Extract only the correction part after the prompt
            if "TESTO CORRETTO:" in generated:
                correction = generated.split("TESTO CORRETTO:")[-1].strip()
                return correction
            else:
                # Fallback: return everything after the original prompt
                return generated[len(prompt):].strip()
        
        elif model_type == "direct":
            model, tokenizer = model_obj
            inputs = tokenizer(prompt, return_tensors="pt")
            
            import torch
            with torch.no_grad():
                outputs = model.generate(
                    **inputs, 
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.eos_token_id
                )
            
            generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract correction part
            if "TESTO CORRETTO:" in generated_text:
                correction = generated_text.split("TESTO CORRETTO:")[-1].strip()
                return correction
            else:
                return generated_text[len(prompt):].strip()
        
    except Exception as e:
        return f"[GENERATION ERROR: {str(e)}] Prompt was: {prompt[:100]}..."

def process_text_with_chunking(model_info, text: str, chunk_size: int = 800) -> str:
    """
    Process long text by splitting into chunks and processing each chunk separately.
    Then combine the results.
    """
    # Split text into manageable chunks
    chunks = split_text_into_chunks(text, max_chars=chunk_size)
    
    if len(chunks) == 1:
        # Single chunk, process normally
        prompt = f"""Correggi SOLO gli errori OCR nel seguente testo italiano, mantenendo ESATTAMENTE lo stesso contenuto e significato. Non aggiungere nulla, non continuare la storia.

TESTO OCR CON ERRORI:
{text}

TESTO CORRETTO:"""
        
        return generate_text_safe(model_info, prompt, max_new_tokens=min(300, len(text) // 2))
    
    # Multiple chunks, process each separately
    print(f"   📄 Processing long text in {len(chunks)} chunks...")
    corrected_chunks = []
    
    for i, chunk in enumerate(chunks):
        print(f"      Processing chunk {i+1}/{len(chunks)} ({len(chunk)} chars)")
        
        prompt = f"""Correggi SOLO gli errori OCR nel seguente testo italiano, mantenendo ESATTAMENTE lo stesso contenuto e significato. Non aggiungere nulla, non continuare la storia.

TESTO OCR CON ERRORI:
{chunk}

TESTO CORRETTO:"""
        
        # Adjust max_new_tokens based on chunk size
        max_tokens = min(300, max(50, len(chunk) // 2))
        corrected_chunk = generate_text_safe(model_info, prompt, max_new_tokens=max_tokens)
        corrected_chunks.append(corrected_chunk)
    
    # Combine chunks back together
    combined_text = " ".join(corrected_chunks)
    
    # Clean up any artifacts from combining
    combined_text = re.sub(r'\s+', ' ', combined_text)  # Remove extra spaces
    combined_text = re.sub(r'\s+([.!?,:;])', r'\1', combined_text)  # Fix punctuation spacing
    
    return combined_text.strip()

def main():
    """Main execution function"""
    print("🚀 Starting Qwen2 OCR Processing Script with Proper Chunking")
    print("🎯 Processing ONLY keys 1-6")
    print("=" * 60)
    
    # Configuration
    TARGET_KEYS = ["1", "2", "3", "4", "5", "6"]  # Only process these keys
    MAX_TOKENS = 10000  # Set to None for no limit, or any number like 5000, 10000, etc.
    CHUNK_SIZE = 800    # Maximum characters per chunk for processing
    
    print(f"🔑 Target keys: {TARGET_KEYS}")
    if MAX_TOKENS:
        print(f"🎯 Token limit: {MAX_TOKENS:,} tokens")
    else:
        print("🎯 No token limit - processing all target keys")
    
    print(f"📏 Chunk size: {CHUNK_SIZE} characters")
    
    # Set up paths
    base_path = Path("./ita")
    if not base_path.exists():
        print(f"❌ Base path {base_path} does not exist")
        print("Please ensure the 'ita' folder with JSON files exists in the current directory")
        return
    
    try:
        # Load data with key and token filtering
        cleaned_data, ocr_data = load_data(base_path, TARGET_KEYS, max_tokens=MAX_TOKENS)
        
        if not ocr_data:
            print("❌ No data found for target keys 1-6!")
            return
        
        # Display sample
        sample_key = list(cleaned_data.keys())[0]
        print("\n📋 Sample Data Preview:")
        pprint({
            "key": sample_key,
            "cleaned (first 200 chars)": cleaned_data[sample_key][:200],
            "ocr     (first 200 chars)": ocr_data[sample_key][:200]
        })
        
        # Preprocess OCR data
        print("\n🔄 Preprocessing OCR data...")
        ocr_cleaned_data = {k: clean_ocr_text(v) for k, v in ocr_data.items()}
        print(f"✅ Preprocessed {len(ocr_cleaned_data)} OCR samples")
        
        # Show chunking statistics
        total_chunks = 0
        long_texts = 0
        for key, text in ocr_cleaned_data.items():
            chunks = split_text_into_chunks(text, CHUNK_SIZE)
            total_chunks += len(chunks)
            if len(chunks) > 1:
                long_texts += 1
                print(f"   📄 Key '{key}': {len(chunks)} chunks ({len(text)} chars)")
        
        print(f"📊 Chunking Statistics:")
        print(f"   • Texts requiring chunking: {long_texts}/{len(ocr_cleaned_data)}")
        print(f"   • Total chunks to process: {total_chunks}")
        print(f"   • Average chunks per text: {total_chunks/len(ocr_cleaned_data):.1f}")
        
        # Initialize model
        model_info = initialize_model_safe()
        
        if model_info[0] is None:
            print("\n⚠️ Model loading failed. Creating outputs with preprocessing only...")
            # Just use the cleaned OCR data as output
            output_dict = ocr_cleaned_data
        else:
            print("\n🔄 Processing OCR samples with Qwen2 model using chunking...")
            output_dict = {}
            
            # Process keys in order 1-6
            sorted_keys = sorted(ocr_cleaned_data.keys(), key=lambda x: int(x))
            total_samples = len(sorted_keys)
            
            for i, k in enumerate(sorted_keys):
                sample_text = ocr_cleaned_data[k]
                print(f"Processing sample {i+1}/{total_samples} (Key: {k}, {len(sample_text)} chars)")
                
                # Process with chunking
                corrected_text = process_text_with_chunking(model_info, sample_text, CHUNK_SIZE)
                output_dict[k] = corrected_text
        
        # Save results
        suffix = f"-keys1to6"
        if MAX_TOKENS:
            suffix += f"-{MAX_TOKENS}tokens"
        else:
            suffix += "-full"
            
        output_path = base_path / f"4gain-hw2_ocr-Qwen2-1.5B-chunked{suffix}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Saved outputs to {output_path}")
        
        # Show a sample result
        sample_key = sorted(output_dict.keys(), key=lambda x: int(x))[0]
        print(f"\n📝 Sample Result (Key: {sample_key}):")
        print("Original OCR (first 400 chars):")
        print(repr(ocr_data[sample_key][:400]) + "...")
        print("\nChunked Qwen2 output (first 400 chars):")
        print(repr(output_dict[sample_key][:400]) + "...")
        print(f"\nOriginal length: {len(ocr_data[sample_key])} chars")
        print(f"Output length: {len(output_dict[sample_key])} chars")
        
        # Show final statistics
        print(f"\n📊 Final Processing Statistics:")
        print(f"   • Keys processed: {sorted(output_dict.keys(), key=lambda x: int(x))}")
        print(f"   • Total samples processed: {len(output_dict)}")
        if model_info[0]:
            avg_orig_len = sum(len(v) for v in ocr_cleaned_data.values()) / len(ocr_cleaned_data)
            avg_corr_len = sum(len(v) for v in output_dict.values()) / len(output_dict)
            print(f"   • Average original length: {avg_orig_len:.0f} characters")
            print(f"   • Average corrected length: {avg_corr_len:.0f} characters")
            print(f"   • Length retention: {(avg_corr_len/avg_orig_len)*100:.1f}%")
        
        print("\n🎉 Script completed with chunking for keys 1-6!")
        print("📝 Long texts have been processed in chunks and recombined")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()