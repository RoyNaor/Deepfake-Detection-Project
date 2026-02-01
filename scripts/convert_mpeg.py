import os
import sys
import torchaudio
import torch

def convert_to_wav(mpeg_path):
    if not os.path.exists(mpeg_path):
        print(f"❌ File not found: {mpeg_path}")
        return

    print(f"Processing: {mpeg_path}...")
    
    # 1. טעינת ה-MPEG (אודיו מתוך קובץ הוידאו/אודיו)
    try:
        waveform, sr = torchaudio.load(mpeg_path)
    except Exception as e:
        print(f"Error loading MPEG. Make sure ffmpeg is installed. Error: {e}")
        return

    # 2. המרה ל-16kHz (קריטי למודל)
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        waveform = resampler(waveform)

    # 3. שמירה כ-WAV
    # שימוש ב-splitext מחליף את הסיומת בצורה בטוחה יותר מאשר replace
    base_name, _ = os.path.splitext(mpeg_path)
    new_path = base_name + ".wav"
    
    torchaudio.save(new_path, waveform, 16000)
    print(f"✅ Converted to: {new_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python convert_mpeg.py file1.mpeg file2.mpeg ...")
    else:
        for f in sys.argv[1:]:
            convert_to_wav(f)