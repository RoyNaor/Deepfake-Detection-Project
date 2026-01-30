import torch
import torchaudio
import transformers

print("--- System Check ---")
print(f"PyTorch Version: {torch.__version__}")
print(f"Torchaudio Version: {torchaudio.__version__}")

# בדיקת GPU
if torch.cuda.is_available():
    print(f"✅ GPU is good: {torch.cuda.get_device_name(0)}")
else:
    print("❌ אזהרה: GPU לא מזוהה! האימון יהיה איטי מאוד.")
    print("נסה להתקין מחדש את PyTorch עם תמיכת CUDA.")

try:
    from transformers import WhisperModel
    print("✅ Transformers is ready")
except ImportError:
    print("❌ שגיאה בטעינת Transformers.")