from transformers import WavLMModel, WhisperModel
import torch

print("--- מתחיל הורדת מודלים (זה ייקח כמה דקות) ---")

print("1. מוריד את Microsoft WavLM Base Plus...")
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
print("✅ WavLM ירד בהצלחה.")

print("2. מוריד את OpenAI Whisper Small...")
whisper = WhisperModel.from_pretrained("openai/whisper-small")
print("✅ Whisper ירד בהצלחה.")

print("--- הכל מוכן! ---")