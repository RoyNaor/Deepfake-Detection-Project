"""Download and cache pretrained WavLM and Whisper models for offline use."""

from transformers import WavLMModel, WhisperModel
import torch

print("--- Starting model downloads (this may take a few minutes) ---")

print("1. Downloading Microsoft WavLM Base Plus...")
# This call stores the model in the local Transformers cache.
wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
print("✅ WavLM downloaded successfully.")

print("2. Downloading OpenAI Whisper Small...")
whisper = WhisperModel.from_pretrained("openai/whisper-small")
print("✅ Whisper downloaded successfully.")

print("--- All models are ready. ---")
