"""Extract WavLM and Whisper features from raw audio files."""

import torch
import torchaudio
from transformers import WavLMModel, WhisperModel, AutoFeatureExtractor
import os
from tqdm import tqdm
import numpy as np

# --- Path configuration ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")
FEATS_WAVLM_DIR = os.path.join(BASE_DIR, "data", "feats_wavlm")
FEATS_WHISPER_DIR = os.path.join(BASE_DIR, "data", "feats_whisper")

# Device selection for model inference.
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def extract_for_folder(model_wavlm, model_whisper, processor, subfolder):
    """Extract features for all audio files in a split folder.

    Args:
        model_wavlm: Pretrained WavLM model (in eval mode).
        model_whisper: Pretrained Whisper model (in eval mode).
        processor: Whisper feature extractor for log-mel spectrograms.
        subfolder: Folder name under RAW_AUDIO_DIR ("train" or "test").

    Side Effects:
        - Writes .pt feature tensors into FEATS_WAVLM_DIR and FEATS_WHISPER_DIR.
    """
    input_dir = os.path.join(RAW_AUDIO_DIR, subfolder)
    out_wavlm = os.path.join(FEATS_WAVLM_DIR, subfolder)
    out_whisper = os.path.join(FEATS_WHISPER_DIR, subfolder)
    
    os.makedirs(out_wavlm, exist_ok=True)
    os.makedirs(out_whisper, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.flac')]
    
    print(f"\n📂 Processing folder: {subfolder} ({len(files)} files)...")
    
    for fname in tqdm(files):
        file_path = os.path.join(input_dir, fname)
        save_name = fname.replace('.flac', '.pt')

        # Skip computation if features are already saved.
        if os.path.exists(os.path.join(out_wavlm, save_name)) and \
           os.path.exists(os.path.join(out_whisper, save_name)):
            continue

        try:
            # 1) Load audio.
            waveform, sr = torchaudio.load(file_path)

            # 2) Resample to 16 kHz for both models.
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)

            # Skip nearly empty clips (<0.1 sec) to avoid invalid features.
            if waveform.shape[1] < 1600:
                continue

            # --- WavLM preparation (truncated raw audio) ---
            # Truncate long clips to save memory; 4 seconds is enough for this POC.
            max_raw_len = 16000 * 4 
            wavlm_input = waveform
            if wavlm_input.shape[1] > max_raw_len:
                wavlm_input = wavlm_input[:, :max_raw_len]

            # Ensure batch dimension.
            if wavlm_input.dim() == 1:
                wavlm_input = wavlm_input.unsqueeze(0)

            # --- Whisper preparation (log-mel spectrograms) ---
            # The processor handles padding/truncation to Whisper's expected length.
            whisper_inputs = processor(
                waveform.squeeze().numpy(), 
                sampling_rate=16000, 
                return_tensors="pt"
            )
            whisper_input_features = whisper_inputs.input_features

            # Move tensors to the selected device for inference.
            wavlm_input = wavlm_input.to(DEVICE)
            whisper_input_features = whisper_input_features.to(DEVICE)

            with torch.no_grad():
                # A) WavLM feature extraction.
                wavlm_out = model_wavlm(wavlm_input).last_hidden_state 
                torch.save(wavlm_out.cpu(), os.path.join(out_wavlm, save_name))

                # B) Whisper feature extraction.
                whisper_out = model_whisper.encoder(whisper_input_features).last_hidden_state 
                torch.save(whisper_out.cpu(), os.path.join(out_whisper, save_name))

        except Exception as e:
            print(f"❌ Error processing file {fname}: {e}")

def main():
    """Entry point that loads models and processes train/test folders."""
    print(f"🚀 Starting feature extraction on {DEVICE}")

    print("Loading WavLM...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE)
    wavlm.eval()

    print("Loading Whisper & Processor...")
    whisper = WhisperModel.from_pretrained("openai/whisper-small").to(DEVICE)
    whisper.eval()
    # The processor converts raw audio into Whisper-compatible spectrograms.
    processor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    
    if os.path.exists(os.path.join(RAW_AUDIO_DIR, "train")):
        extract_for_folder(wavlm, whisper, processor, "train")
    else:
        print("⚠️ Train folder not found!")

    if os.path.exists(os.path.join(RAW_AUDIO_DIR, "test")):
        extract_for_folder(wavlm, whisper, processor, "test")
        
    print("\n✅✅✅ Finished! All features extracted.")

if __name__ == "__main__":
    main()
