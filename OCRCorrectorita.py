#!/usr/bin/env python3
"""
OCR Text Processing Script using DeepMount00/OCR_corrector (Italian OCR Error Correction)
Modified to process specific chunks from "1" to "6"
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

def filter_chunks_1_to_6(data_dict: dict) -> dict:
    """Filter dataset to include only chunks "1" through "6" """
    filtered_data = {}
    target_keys = ["1", "2", "3", "4", "5", "6"]
    
    print(f"🎯 Filtering dataset to chunks 1-6...")
    
    for key in target_keys:
        if key in data_dict:
            filtered_data[key] = data_dict[key]
            print(f"   ✅ Found chunk {key}: {len(data_dict[key])} characters")
        else:
            print(f"   ⚠️ Chunk {key} not found in dataset")
    
    total_tokens = sum(estimate_tokens(text) for text in filtered_data.values())
    print(f"✅ Selected {len(filtered_data)} chunks (~{total_tokens:,} tokens)")
    return filtered_data

def load_data(base_path: Path, use_chunks_1_to_6: bool = True):
    """Load cleaned and OCR data from JSON files with chunk filtering"""
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
    
    # Show available keys for debugging
    print(f"📋 Available keys in dataset: {sorted(list(ocr_data.keys())[:10])}{'...' if len(ocr_data) > 10 else ''}")
    
    # Apply chunk filtering if specified
    if use_chunks_1_to_6:
        ocr_data = filter_chunks_1_to_6(ocr_data)
        # Filter cleaned_data to match the same keys
        cleaned_data = {k: v for k, v in cleaned_data.items() if k in ocr_data}
        print(f"📊 After filtering: {len(ocr_data)} chunks")
    
    return cleaned_data, ocr_data

def try_import_transformers():
    """Try to import transformers with better error handling"""
    try:
        print("🔍 Checking transformers installation...")
        
        # Try basic import first
        import transformers
        print(f"✅ Transformers version: {transformers.__version__}")
        
        # Check if version is sufficient for seq2seq models
        version_parts = transformers.__version__.split('.')
        major, minor = int(version_parts[0]), int(version_parts[1])
        
        if major < 4:
            print(f"⚠️ Warning: Transformers version {transformers.__version__} may be outdated")
            print("   Recommended: >= 4.0.0")
        
        return transformers
        
    except ImportError as e:
        print(f"❌ Cannot import transformers: {e}")
        print("\n🔧 Please install transformers:")
        print("   pip install transformers")
        return None
    except Exception as e:
        print(f"❌ Error with transformers: {e}")
        return None

def check_torch_and_device():
    """Check PyTorch installation and available device"""
    try:
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda":
            print(f"🚀 CUDA available - will use GPU (Device: {torch.cuda.get_device_name()})")
        else:
            print("💻 CUDA not available - will use CPU")
        return torch, device
    except ImportError:
        print("❌ PyTorch not found!")
        print("   Please install: pip install torch")
        return None, "cpu"

def initialize_ocr_corrector():
    """Initialize OCR Corrector model with safe error handling"""
    transformers = try_import_transformers()
    if not transformers:
        return None, None, None
    
    torch, device = check_torch_and_device()
    if not torch:
        return None, None, None
    
    print("\n🤖 Initializing DeepMount00/OCR_corrector model...")
    print("💾 Note: Model will be downloaded only once (cached)")
    
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        
        MODEL_NAME = "DeepMount00/OCR_corrector"
        
        print(f"📝 Loading tokenizer from {MODEL_NAME}...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        
        print(f"📝 Loading model from {MODEL_NAME}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).eval()
        
        print(f"📝 Moving model to device: {device}")
        model.to(device)
        
        print("✅ OCR Corrector model loaded successfully!")
        print(f"📊 Model loaded on: {device}")
        
        return model, tokenizer, device
        
    except Exception as e:
        print(f"❌ OCR Corrector loading failed: {e}")
        
        print("\n🔧 Troubleshooting suggestions:")
        print("1. Create a virtual environment:")
        print("   python -m venv ocr_env")
        print("   source ocr_env/bin/activate  # On Windows: ocr_env\\Scripts\\activate")
        print("2. Install packages:")
        print("   pip install torch transformers")
        print("3. Check internet connection (model needs to be downloaded)")
        print("4. Try running the script again")
        
        return None, None, None

def correct_ocr_text(model, tokenizer, device, ocr_text: str, max_length: int = 1050):
    """Correct OCR text using the specialized OCR corrector model"""
    if not model or not tokenizer:
        return f"[MODEL NOT AVAILABLE] {ocr_text[:100]}..."
    
    try:
        import torch  # Import torch here to ensure it's available
        
        # Tokenize input text
        inputs = tokenizer(ocr_text, return_tensors="pt").to(device)
        
        # Generate corrected text
        with torch.no_grad():  # Disable gradient computation for inference
            outputs = model.generate(
                input_ids=inputs['input_ids'],
                attention_mask=inputs['attention_mask'],
                num_beams=2,          # Use beam search for better quality
                max_length=max_length, # Allow up to 1050 tokens as per example
                top_k=10,             # Limit vocabulary at each step
                do_sample=False,      # Use deterministic generation
                early_stopping=True   # Stop when end token is generated
            )
        
        # Decode the corrected text
        corrected_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        return corrected_text.strip()
        
    except Exception as e:
        print(f"⚠️ Error correcting text: {str(e)}")
        # Fallback to basic OCR cleaning
        return clean_ocr_text(ocr_text)

def split_long_text(text: str, max_chars: int = 800) -> list:
    """Split long text into smaller chunks for processing"""
    if len(text) <= max_chars:
        return [text]
    
    chunks = []
    sentences = re.split(r'[.!?]+', text)
    current_chunk = ""
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        # If adding this sentence would exceed limit, save current chunk
        if len(current_chunk) + len(sentence) > max_chars and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += ". " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks

def process_long_text(model, tokenizer, device, ocr_text: str) -> str:
    """Process long text by splitting into chunks and recombining"""
    # Split text into manageable chunks
    chunks = split_long_text(ocr_text, max_chars=800)
    
    if len(chunks) == 1:
        return correct_ocr_text(model, tokenizer, device, ocr_text)
    
    print(f"   📄 Processing long text in {len(chunks)} chunks...")
    corrected_chunks = []
    
    for i, chunk in enumerate(chunks):
        corrected_chunk = correct_ocr_text(model, tokenizer, device, chunk)
        corrected_chunks.append(corrected_chunk)
    
    return " ".join(corrected_chunks)

def main():
    """Main execution function"""
    print("🚀 Starting OCR Processing with DeepMount00/OCR_corrector")
    print("=" * 65)
    print("🎯 This model corrects ~93% of Italian OCR errors")
    print("📋 Processing chunks 1-6 specifically")
    
    # Set up paths
    base_path = Path("./ita")
    if not base_path.exists():
        print(f"❌ Base path {base_path} does not exist")
        print("Please ensure the 'ita' folder with JSON files exists in the current directory")
        return
    
    try:
        # Load data with chunk filtering (chunks 1-6)
        cleaned_data, ocr_data = load_data(base_path, use_chunks_1_to_6=True)
        
        if not ocr_data:
            print("❌ No chunks 1-6 found in the dataset!")
            return
        
        # Display sample
        sample_key = list(cleaned_data.keys())[0]
        print(f"\n📋 Sample Data Preview (Chunk {sample_key}):")
        pprint({
            "cleaned (first 200 chars)": cleaned_data[sample_key][:200],
            "ocr     (first 200 chars)": ocr_data[sample_key][:200]
        })
        
        # Show all chunks that will be processed
        print(f"\n📊 Chunks to be processed:")
        for key in sorted(ocr_data.keys(), key=lambda x: int(x)):
            char_count = len(ocr_data[key])
            token_estimate = estimate_tokens(ocr_data[key])
            print(f"   • Chunk {key}: {char_count:,} characters (~{token_estimate:,} tokens)")
        
        # Preprocess OCR data
        print("\n🔄 Preprocessing OCR data...")
        ocr_cleaned_data = {k: clean_ocr_text(v) for k, v in ocr_data.items()}
        print(f"✅ Preprocessed {len(ocr_cleaned_data)} OCR samples")
        
        # Initialize OCR Corrector model
        model, tokenizer, device = initialize_ocr_corrector()
        
        if model is None:
            print("\n⚠️ OCR Corrector model loading failed. Using basic OCR cleaning only...")
            # Just use the cleaned OCR data as output
            output_dict = ocr_cleaned_data
        else:
            print("\n🔄 Processing OCR samples with OCR Corrector...")
            output_dict = {}
            
            # Process chunks in order
            for chunk_key in sorted(ocr_cleaned_data.keys(), key=lambda x: int(x)):
                sample_text = ocr_cleaned_data[chunk_key]
                print(f"Processing chunk {chunk_key}...")
                
                # Process the text (handles long texts automatically)
                corrected_text = process_long_text(model, tokenizer, device, sample_text)
                output_dict[chunk_key] = corrected_text
        
        # Save results
        output_path = base_path / "4gain-hw2_ocr-OCRCorrector-cleaned-chunks1to6.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_dict, f, ensure_ascii=False, indent=2)
        
        print(f"\n✅ Saved outputs to {output_path}")
        
        # Show a sample result
        sample_key = list(output_dict.keys())[0]
        print(f"\n📝 Sample Result (Chunk {sample_key}):")
        print("Original OCR (first 300 chars):")
        print(ocr_data[sample_key][:300] + "...")
        print("\nOCR Corrector output (first 300 chars):")
        print(output_dict[sample_key][:300] + "...")
        
        # Show some statistics
        print(f"\n📊 Processing Statistics:")
        print(f"   • Total chunks processed: {len(output_dict)}")
        print(f"   • Chunks processed: {', '.join(sorted(output_dict.keys(), key=lambda x: int(x)))}")
        if model:
            avg_orig_len = sum(len(v) for v in ocr_cleaned_data.values()) / len(ocr_cleaned_data)
            avg_corr_len = sum(len(v) for v in output_dict.values()) / len(output_dict)
            print(f"   • Average original length: {avg_orig_len:.0f} characters")
            print(f"   • Average corrected length: {avg_corr_len:.0f} characters")
        
        total_chars = sum(len(v) for v in output_dict.values())
        total_tokens = sum(estimate_tokens(v) for v in output_dict.values())
        print(f"   • Total characters processed: {total_chars:,}")
        print(f"   • Total tokens processed: {total_tokens:,}")
        
        print("\n🎉 Script completed!")
        print("📈 OCR Corrector model typically achieves ~93% error correction rate")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()