import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys
import copy

# הוספת התיקייה הראשית ל-Path כדי שנוכל לייבא מודלים
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.nes2net import Nes2Net

# --- הגדרות ---
DATA_ROOT = "../data"
PROTOCOLS_DIR = os.path.join(DATA_ROOT, "protocols")
FEATS_WAVLM = os.path.join(DATA_ROOT, "feats_wavlm")
FEATS_WHISPER = os.path.join(DATA_ROOT, "feats_whisper")
RESULTS_DIR = "../results"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# פרמטרים לאימון
BATCH_SIZE = 16 # אם יש לך GPU חזק, אפשר להגדיל ל-32
EPOCHS = 20     # מספיק ל-POC
LR = 0.0001     # קצב למידה

# --- 1. Dataset Class חכם ---
class FeatureDataset(Dataset):
    def __init__(self, mode, split='train'):
        """
        mode: 'wavlm', 'whisper', 'fusion'
        split: 'train' or 'test'
        """
        self.mode = mode
        self.split = split
        
        # טעינת הפרוטוקול (רשימת הקבצים והתוויות)
        protocol_file = os.path.join(PROTOCOLS_DIR, f"{split}_protocol.txt")
        self.file_list = []
        self.labels = []
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                fname = parts[0]
                label_str = parts[1] # 'bonafide' or 'spoof'
                
                self.file_list.append(fname)
                # המרה למספרים: bonafide (אמיתי) = 1, spoof (מזויף) = 0
                self.labels.append(1 if label_str == 'bonafide' else 0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        label = self.labels[idx]
        
        # נתיבים לקבצי הפיצ'רים
        # (אנחנו מניחים שההרחבה היא .pt כי חילצנו אותם קודם)
        wavlm_path = os.path.join(FEATS_WAVLM, fname + ".pt")
        whisper_path = os.path.join(FEATS_WHISPER, fname + ".pt")
        
        # טעינה לפי המוד (Mode)
        try:
            if self.mode == 'wavlm':
                feat = torch.load(wavlm_path) # [1, Time, 768]
                
            elif self.mode == 'whisper':
                feat = torch.load(whisper_path) # [1, Time, 768]
                
            elif self.mode == 'fusion':
                feat_w = torch.load(wavlm_path)     # WavLM
                feat_s = torch.load(whisper_path)   # Whisper
                
                # --- סנכרון (Alignment) ---
                # Whisper ו-WavLM באורכים שונים. נסנכרן לפי WavLM.
                target_len = feat_w.shape[1]
                
                # מתיחת Whisper
                feat_s = feat_s.transpose(1, 2) # [1, 768, Time]
                feat_s = torch.nn.functional.interpolate(feat_s, size=target_len, mode='linear')
                feat_s = feat_s.transpose(1, 2) # [1, Time, 768]
                
                # שרשור (Concatenation)
                feat = torch.cat((feat_w, feat_s), dim=2) # [1, Time, 1536]
                
        except FileNotFoundError as e:
            # במקרה שקובץ חסר (לא אמור לקרות אם חילצת הכל)
            print(f"Error loading {fname}: {e}")
            return torch.zeros(768, 400), label

        # עיבוד סופי ל-Nes2Net
        # Nes2Net רוצה: [Channels, Time] וללא מימד ה-Batch הראשון של ה-Loader
        feat = feat.squeeze(0).transpose(0, 1) # [Channels, Time]
        
        # חיתוך או ריפוד לאורך קבוע (כדי שנוכל לעשות Batch)
        max_len = 400
        if feat.shape[1] > max_len:
            feat = feat[:, :max_len]
        else:
            padding = max_len - feat.shape[1]
            feat = torch.nn.functional.pad(feat, (0, padding))
            
        return feat, label

# --- 2. פונקציות אימון ובדיקה ---
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for feats, labels in loader:
        feats, labels = feats.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        outputs = model(feats) # Nes2Net output
        
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
    return total_loss / len(loader), 100 * correct / total

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    return total_loss / len(loader), 100 * correct / total

# --- 3. המנוע הראשי שמריץ את הניסויים ---
def run_experiment(mode_name):
    print(f"\n{'='*40}")
    print(f"🚀 מתחיל ניסוי: {mode_name.upper()}")
    print(f"{'='*40}")
    
    # הגדרת גודל הקלט
    input_dim = 1536 if mode_name == 'fusion' else 768
    
    # אתחול המודל (Nes2Net הנקי)
    model = Nes2Net(input_channels=input_dim).to(DEVICE)
    
    # אתחול DataLoaders
    train_ds = FeatureDataset(mode=mode_name, split='train')
    test_ds = FeatureDataset(mode=mode_name, split='test')
    
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    best_acc = 0.0
    
    for epoch in range(EPOCHS):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion)
        test_loss, test_acc = evaluate(model, test_loader, criterion)
        
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}%")
        
        # שמירת המודל הכי טוב
        if test_acc > best_acc:
            best_acc = test_acc
            # שמירת המשקולות
            os.makedirs(RESULTS_DIR, exist_ok=True)
            torch.save(model.state_dict(), f"{RESULTS_DIR}/best_model_{mode_name}.pth")
            
    print(f"🏁 סיכום ניסוי {mode_name}: Best Test Accuracy = {best_acc:.2f}%")
    return best_acc

# --- 4. הרצה ראשית ---
if __name__ == "__main__":
    if not os.path.exists(FEATS_WAVLM):
        print("❌ שגיאה: לא נמצאו פיצ'רים מחולצים. אנא הריצי קודם את extract_features.py")
        exit()

    results = {}
    
    # ניסוי 1: WavLM בלבד (Baseline)
    results['WavLM'] = run_experiment('wavlm')
    
    # ניסוי 2: Whisper בלבד (Semantics)
    results['Whisper'] = run_experiment('whisper')
    
    # ניסוי 3: Fusion (המודל שלכם)
    results['Fusion'] = run_experiment('fusion')
    
    # הדפסת טבלת סיכום
    print("\n\n📊 --- טבלת תוצאות סופית --- 📊")
    print(f"{'Model':<15} | {'Accuracy':<10}")
    print("-" * 30)
    for name, acc in results.items():
        print(f"{name:<15} | {acc:.2f}%")
    
    print("-" * 30)
    if results['Fusion'] > results['WavLM']:
        print("✅ הצלחה! מודל ה-Fusion השיג תוצאה טובה יותר מהבסיס.")
    else:
        print("⚠️ שים לב: ה-Fusion לא שיפר את התוצאה. נדרש כוונון נוסף.")