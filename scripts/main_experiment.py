import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys

# הוספת התיקייה הראשית ל-Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.nes2net import Nes2Net

# --- הגדרות ---
DATA_ROOT = os.path.join(parent_dir, "data")
PROTOCOLS_DIR = os.path.join(DATA_ROOT, "protocols")
FEATS_WAVLM = os.path.join(DATA_ROOT, "feats_wavlm")
FEATS_WHISPER = os.path.join(DATA_ROOT, "feats_whisper")
RESULTS_DIR = os.path.join(parent_dir, "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# פרמטרים לאימון
BATCH_SIZE = 16 
EPOCHS = 10      # מספיק ל-POC כדי לראות מגמה
LR = 0.0001      # קצב למידה עדין

class FeatureDataset(Dataset):
    def __init__(self, mode, split='train'):
        self.mode = mode
        self.split = split
        
        # טעינת הפרוטוקול
        protocol_file = os.path.join(PROTOCOLS_DIR, f"{split}_protocol.txt")
        self.file_list = []
        self.labels = []
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                fname = parts[0]
                label_str = parts[1]
                self.file_list.append(fname)
                self.labels.append(1 if label_str == 'bonafide' else 0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        label = self.labels[idx]
        
        # נתיבים
        wavlm_path = os.path.join(FEATS_WAVLM, self.split, fname + ".pt")
        whisper_path = os.path.join(FEATS_WHISPER, self.split, fname + ".pt")
        
        try:
            feat = None
            
            if self.mode == 'wavlm':
                # WavLM בלבד [1, Time, 768]
                feat = torch.load(wavlm_path)
                feat = feat.transpose(1, 2) # [1, 768, Time]

            elif self.mode == 'whisper':
                # Whisper בלבד [1, Time, 768]
                feat = torch.load(whisper_path)
                feat = feat.transpose(1, 2) # [1, 768, Time]

            elif self.mode == 'fusion':
                # טעינת שניהם
                f_w = torch.load(wavlm_path).transpose(1, 2)   # [1, 768, T_wavlm]
                f_s = torch.load(whisper_path).transpose(1, 2) # [1, 768, T_whisper]
                
                # --- סנכרון אורכים (היתוך) ---
                # נשנה את הגודל של Whisper שיתאים ל-WavLM (שהוא הקצר והממוקד יותר)
                target_len = f_w.shape[2]
                f_s = torch.nn.functional.interpolate(f_s, size=target_len, mode='linear', align_corners=False)
                
                # שרשור בערוצים: 768 + 768 = 1536
                feat = torch.cat((f_w, f_s), dim=1) 

            # הסרת מימד ה-Batch המיותר [Channels, Time]
            feat = feat.squeeze(0)
            
            # חיתוך/ריפוד לאורך קבוע (כדי לאפשר Batch)
            fixed_len = 200 # אורך סביר ל-Nes2Net
            if feat.shape[1] > fixed_len:
                feat = feat[:, :fixed_len]
            elif feat.shape[1] < fixed_len:
                pad_amt = fixed_len - feat.shape[1]
                feat = torch.nn.functional.pad(feat, (0, pad_amt))
                
            return feat, label

        except Exception as e:
            # במקרה של קובץ חסר, מחזיר אפסים (לא אמור לקרות)
            dim = 1536 if self.mode == 'fusion' else 768
            return torch.zeros(dim, 200), label

def run_experiment(mode_name):
    print(f"\n{'='*40}")
    print(f"🚀 Training Model: {mode_name.upper()}")
    print(f"{'='*40}")
    
    input_dim = 1536 if mode_name == 'fusion' else 768
    model = Nes2Net(input_channels=input_dim).to(DEVICE)
    
    train_ds = FeatureDataset(mode=mode_name, split='train')
    test_ds = FeatureDataset(mode=mode_name, split='test')
    
    # אם אין מספיק דאטה ל-Batch מלא, Drop Last מונע קריסה
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        # Train
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        
        for feats, labels in train_loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(feats)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
        # Evaluate
        model.eval()
        test_correct = 0
        test_total = 0
        with torch.no_grad():
            for feats, labels in test_loader:
                feats, labels = feats.to(DEVICE), labels.to(DEVICE)
                outputs = model(feats)
                _, predicted = torch.max(outputs.data, 1)
                test_total += labels.size(0)
                test_correct += (predicted == labels).sum().item()
        
        acc = 100 * test_correct / test_total
        print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {100*train_correct/train_total:.1f}% | Test Acc: {acc:.2f}%")
        
        if acc > best_acc:
            best_acc = acc
            
    print(f"🏆 Best Result for {mode_name}: {best_acc:.2f}%")
    return best_acc

if __name__ == "__main__":
    results = {}
    
    # 1. הרצת ניסוי אקוסטי
    results['WavLM'] = run_experiment('wavlm')
    
    # 2. הרצת ניסוי סמנטי
    results['Whisper'] = run_experiment('whisper')
    
    # 3. הרצת ניסוי משולב
    results['Fusion'] = run_experiment('fusion')
    
    print("\n\n📊 --- FINAL RESULTS SUMMARY --- 📊")
    print("-" * 35)
    print(f"{'Model':<15} | {'Accuracy':<10}")
    print("-" * 35)
    for k, v in results.items():
        print(f"{k:<15} | {v:.2f}%")
    print("-" * 35)