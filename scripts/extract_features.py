import torch
import torchaudio
from transformers import WavLMModel, WhisperModel
import os
from tqdm import tqdm

# --- הגדרות נתיבים ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # תיקיית הפרויקט הראשית
RAW_AUDIO_DIR = os.path.join(BASE_DIR, "data", "raw_audio")
FEATS_WAVLM_DIR = os.path.join(BASE_DIR, "data", "feats_wavlm")
FEATS_WHISPER_DIR = os.path.join(BASE_DIR, "data", "feats_whisper")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def extract_for_folder(model_wavlm, model_whisper, subfolder):
    """
    פונקציה שמקבלת שם של תת-תיקייה (train או test) ומחלצת את כל הקבצים שבתוכה
    """
    input_dir = os.path.join(RAW_AUDIO_DIR, subfolder)
    
    # יצירת תיקיות יעד תואמות
    out_wavlm = os.path.join(FEATS_WAVLM_DIR, subfolder)
    out_whisper = os.path.join(FEATS_WHISPER_DIR, subfolder)
    
    os.makedirs(out_wavlm, exist_ok=True)
    os.makedirs(out_whisper, exist_ok=True)
    
    # רשימת כל קבצי האודיו
    files = [f for f in os.listdir(input_dir) if f.endswith('.flac')]
    
    print(f"\n📂 מעבד את תיקיית: {subfolder} ({len(files)} קבצים)...")
    
    for fname in tqdm(files):
        file_path = os.path.join(input_dir, fname)
        save_name = fname.replace('.flac', '.pt')
        
        # בדיקה אם כבר עשינו את הקובץ הזה (כדי לחסוך זמן אם עוצרים באמצע)
        if os.path.exists(os.path.join(out_wavlm, save_name)) and \
           os.path.exists(os.path.join(out_whisper, save_name)):
            continue

        try:
            # 1. טעינת אודיו
            waveform, sr = torchaudio.load(file_path)
            
            # Resampling חובה ל-16kHz (שני המודלים דורשים את זה)
            if sr != 16000:
                waveform = torchaudio.functional.resample(waveform, sr, 16000)
            
            # הוספת מימד Batch אם חסר (צריך להיות [1, Time])
            if waveform.dim() == 1:
                waveform = waveform.unsqueeze(0)
            
            waveform = waveform.to(DEVICE)
            
            with torch.no_grad():
                # --- A. חילוץ WavLM ---
                # WavLM מצפה ל-Raw Audio
                wavlm_out = model_wavlm(waveform).last_hidden_state # [1, Time, 768]
                torch.save(wavlm_out.cpu(), os.path.join(out_wavlm, save_name))
                
                # --- B. חילוץ Whisper ---
                # Whisper מצפה גם ל-Raw Audio (בשימוש ב-Feature Extractor הפנימי שלו)
                # נשתמש ישירות ב-Encoder
                whisper_out = model_whisper.encoder(waveform).last_hidden_state # [1, Time, 768]
                torch.save(whisper_out.cpu(), os.path.join(out_whisper, save_name))
                
        except Exception as e:
            print(f"❌ שגיאה בקובץ {fname}: {e}")

def main():
    print(f"🚀 מתחיל חילוץ מאפיינים על {DEVICE}")
    
    # 1. טעינת המודלים הכבדים לזיכרון (פעם אחת בלבד)
    print("Loading WavLM...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE)
    wavlm.eval() # מצב קריאה בלבד
    
    print("Loading Whisper...")
    whisper = WhisperModel.from_pretrained("openai/whisper-small").to(DEVICE)
    whisper.eval()
    
    # 2. הרצה על תיקיית האימון
    if os.path.exists(os.path.join(RAW_AUDIO_DIR, "train")):
        extract_for_folder(wavlm, whisper, "train")
    else:
        print("⚠️ לא מצאתי תיקיית train! האם הרצת את organize_data.py?")

    # 3. הרצה על תיקיית הטסט
    if os.path.exists(os.path.join(RAW_AUDIO_DIR, "test")):
        extract_for_folder(wavlm, whisper, "test")
        
    print("\n✅✅✅ סיימנו! כל המאפיינים חולצו ומוכנים לאימון.")

if __name__ == "__main__":
    main()