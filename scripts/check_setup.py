"""Run a quick environment check for core ML dependencies and GPU availability."""

import torch
import torchaudio
import transformers

print("--- System Check ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchaudio Version: {torchaudio.__version__}")

# GPU availability check to warn about slow CPU-only training.
if torch.cuda.is_available():
    print(f"✅ GPU is good: {torch.cuda.get_device_name(0)}")
else:
    print("❌ Warning: GPU not detected. Training will be significantly slower.")
    print("Reinstall PyTorch with CUDA support to enable GPU acceleration.")

try:
    from transformers import WhisperModel
    print("✅ Transformers is ready")
except ImportError:
    print("❌ Error: Transformers could not be imported.")
