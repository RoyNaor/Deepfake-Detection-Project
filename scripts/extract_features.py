import torch
import torchaudio
from transformers import WavLMModel, WhisperModel, AutoFeatureExtractor
import os
from tqdm import tqdm

# --- Path Configurations ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) 
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")
FEATS_WAVLM_DIR = os.path.join(BASE_DIR, "data", "feats_wavlm")
FEATS_WHISPER_DIR = os.path.join(BASE_DIR, "data", "feats_whisper")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standardizing to 4 seconds for POC stability
MAX_AUDIO_SEC = 4
SAMPLE_RATE = 16000
MAX_SAMPLES = MAX_AUDIO_SEC * SAMPLE_RATE

def extract_for_split(model_wavlm, model_whisper, processor, split_name):
    input_dir = os.path.join(RAW_AUDIO_DIR, split_name)
    out_wavlm = os.path.join(FEATS_WAVLM_DIR, split_name)
    out_whisper = os.path.join(FEATS_WHISPER_DIR, split_name)
    
    os.makedirs(out_wavlm, exist_ok=True)
    os.makedirs(out_whisper, exist_ok=True)
    
    files = [f for f in os.listdir(input_dir) if f.endswith('.flac')]
    print(f"\n--- Extracting Features for split: {split_name.upper()} ({len(files)} files) ---")
    
    for fname in tqdm(files):
        file_path = os.path.join(input_dir, fname)
        save_name = fname.replace('.flac', '.pt')
        
        # Skip if already processed
        if os.path.exists(os.path.join(out_wavlm, save_name)) and \
           os.path.exists(os.path.join(out_whisper, save_name)):
            continue

        try:
            # 1. Load and Resample
            waveform, sr = torchaudio.load(file_path)
            if waveform.shape[0] > 1: waveform = waveform[:1, :] # To Mono
            if sr != SAMPLE_RATE:
                waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)
            
            # 2. Unified Truncation/Padding (Fixing the 4-second window)
            if waveform.shape[1] > MAX_SAMPLES:
                waveform = waveform[:, :MAX_SAMPLES]
            else:
                pad_amount = MAX_SAMPLES - waveform.shape[1]
                waveform = torch.nn.functional.pad(waveform, (0, pad_amount))

            # 3. Prepare Inputs
            wavlm_input = waveform.to(DEVICE)
            
            # Whisper Processor handles Mel-spectrogram conversion
            whisper_inputs = processor(
                waveform.squeeze().numpy(), 
                sampling_rate=SAMPLE_RATE, 
                return_tensors="pt"
            )
            whisper_input_features = whisper_inputs.input_features.to(DEVICE)
            
            with torch.no_grad():
                # A. Extract WavLM (Acoustic)
                wavlm_out = model_wavlm(wavlm_input).last_hidden_state 
                torch.save(wavlm_out.cpu(), os.path.join(out_wavlm, save_name))
                
                # B. Extract Whisper (Semantic)
                whisper_out = model_whisper.encoder(whisper_input_features).last_hidden_state 
                torch.save(whisper_out.cpu(), os.path.join(out_whisper, save_name))
                
        except Exception as e:
            print(f"Error processing {fname}: {e}")

def main():
    print(f"Starting feature extraction on {DEVICE}...")
    
    # Load Models
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval()
    whisper = WhisperModel.from_pretrained("openai/whisper-small").to(DEVICE).eval()
    processor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")
    
    # Process all existing splits
    for split in ['train', 'dev', 'test']:
        if os.path.exists(os.path.join(RAW_AUDIO_DIR, split)):
            extract_for_split(wavlm, whisper, processor, split)
        
    print("\n✅ Feature extraction completed for all splits.")

if __name__ == "__main__":
    main()