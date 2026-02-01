import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from transformers import WavLMModel, WhisperModel, AutoFeatureExtractor
import argparse
import os
import sys

# --- תיקון נתיבים ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.nes2net import Nes2Net

class InferenceSystem(nn.Module):
    def __init__(self, mode):
        super(InferenceSystem, self).__init__()
        self.mode = mode.lower()
        
        # 1. טעינת המודלים הגדולים
        if self.mode in ['wavlm', 'fusion']:
            print("Loading WavLM...")
            self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
            
        if self.mode in ['whisper', 'fusion']:
            print("Loading Whisper...")
            self.whisper = WhisperModel.from_pretrained("openai/whisper-small")

        # 2. הגדרת ה-Backend
        if self.mode == 'wavlm':
            input_dim = 768
        elif self.mode == 'whisper':
            input_dim = 768
        elif self.mode == 'fusion':
            input_dim = 1536
            
        self.backend = Nes2Net(input_channels=input_dim)

    def forward(self, wav_input, whisper_input=None):
        wavlm_out = None
        whisper_out = None
        
        # A. WavLM (מקבל אודיו גולמי)
        if self.mode in ['wavlm', 'fusion']:
            # wav_input shape: [Batch, Time]
            wavlm_out = self.wavlm(wav_input).last_hidden_state # [B, T, 768]

        # B. Whisper (מקבל ספקטרוגרמה)
        if self.mode in ['whisper', 'fusion']:
            if whisper_input is None:
                raise ValueError("Whisper input is missing!")
            # whisper_input shape: [Batch, 80, 3000]
            whisper_out = self.whisper.encoder(whisper_input).last_hidden_state # [B, Tw, 768]

        # C. Fusion & Preprocessing
        final_feat = None
        
        if self.mode == 'wavlm':
            final_feat = wavlm_out.transpose(1, 2)
            
        elif self.mode == 'whisper':
            final_feat = whisper_out.transpose(1, 2)
            
        elif self.mode == 'fusion':
            # אינטרפולציה
            target_len = wavlm_out.shape[1]
            whisper_transposed = whisper_out.transpose(1, 2)
            
            whisper_aligned = F.interpolate(
                whisper_transposed, size=target_len, mode='linear', align_corners=False
            )
            
            wavlm_transposed = wavlm_out.transpose(1, 2)
            final_feat = torch.cat((wavlm_transposed, whisper_aligned), dim=1) # [B, 1536, T]

        # D. Backend
        score = self.backend(final_feat)
        return score

def run_prediction(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"--- Predicting with mode: {args.mode} ---")
    
    # 1. טעינת המודל וה-Processor של Whisper
    model = InferenceSystem(args.mode).to(device)
    model.eval()
    
    whisper_processor = None
    if args.mode in ['whisper', 'fusion']:
        whisper_processor = AutoFeatureExtractor.from_pretrained("openai/whisper-small")

    # 2. טעינת המשקולות
    if os.path.exists(args.checkpoint):
        print(f"Loading backend weights from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.backend.load_state_dict(checkpoint)
        print("✅ Weights loaded successfully!")
    else:
        print(f"❌ Error: Checkpoint not found at {args.checkpoint}")
        return

    # 3. עיבוד האודיו
    try:
        waveform, sr = torchaudio.load(args.file)
    except Exception as e:
        print(f"❌ Error loading audio: {e}")
        return

    # המרה ל-16kHz
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    
    # המרה ל-Mono
    if waveform.shape[0] > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True) # [1, Total_Time]

    # חיתוך ל-4 שניות (כדי להתאים לאימון ולמנוע קריסה)
    MAX_SECONDS = 4
    max_samples = 16000 * MAX_SECONDS
    if waveform.shape[1] > max_samples:
        waveform = waveform[:, :max_samples]
    
    # הכנת הקלטים
    wav_input = None
    whisper_input = None

    # עבור WavLM: צריך [Batch, Time] -> [1, Time]
    # waveform הוא כרגע [1, Time]. זה מושלם ל-Batch=1.
    wav_input = waveform.to(device)

    # עבור Whisper: צריך לחשב Features
    if args.mode in ['whisper', 'fusion']:
        # המעבד מצפה ל-numpy array שטוח (1D)
        raw_audio = waveform.squeeze().cpu().numpy()
        inputs = whisper_processor(raw_audio, sampling_rate=16000, return_tensors="pt")
        whisper_input = inputs.input_features.to(device) # [1, 80, 3000]

    # 4. הרצה
    with torch.no_grad():
        # שליחת הקלטים המתאימים למודל
        logits = model(wav_input, whisper_input)
        probs = F.softmax(logits, dim=1)
        
        fake_prob = probs[0, 0].item()
        real_prob = probs[0, 1].item()
        
    print("\n" + "="*30)
    print(f"File: {os.path.basename(args.file)}")
    print(f"Probability FAKE: {fake_prob*100:.2f}%")
    print(f"Probability REAL: {real_prob*100:.2f}%")
    
    if fake_prob > 0.5:
        print("🔴 VERDICT: SPOOF (FAKE)")
    else:
        print("🟢 VERDICT: BONAFIDE (REAL)")
    print("="*30 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['wavlm', 'whisper', 'fusion'])
    parser.add_argument("--file", type=str, required=True, help="Path to audio file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth file")
    
    args = parser.parse_args()
    run_prediction(args)