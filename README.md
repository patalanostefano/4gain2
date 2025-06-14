# cerbero env
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install transformers>=4.37.0 accelerate safetensors
pip install datasets tokenizers sentencepiece
pip install bitsandbytes  # For 8-bit/4-bit quantization (GPU only)
