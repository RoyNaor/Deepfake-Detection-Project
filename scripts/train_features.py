import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm
import sys

# --- תיקון הנתיבים (החלק הקריטי שהיה חסר) ---
# אנחנו לוקחים את הנתיב של הקובץ הנוכחי, עולים תיקייה אחת למעלה (ל-POC)
# ומוסיפים את זה לנתיבי החיפוש של פייתון
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../POC/scripts
parent_dir = os.path.dirname(current_dir)              # .../POC
sys.path.append(parent_dir)

# עכשיו הייבוא יעבוד כי הוא מחפש ב-POC
from models.nes2net import Nes2Net 

# --- הגדרות קבועים ונתיבים ---
# BASE_DIR הוא התיקייה שמעל scripts (כלומר POC)
BASE_DIR = parent_dir 
DATA_DIR = os.path.join(BASE_DIR, "data")
PROTOCOLS_DIR = os.path.join(DATA_DIR, "protocols")
FEATS_WAVLM = os.path.join(DATA_DIR, "feats_wavlm")
FEATS_WHISPER = os.path.join(DATA_DIR, "feats_whisper")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# אורך קלט קבוע ל-Nes2Net (מייצג ציר הזמן לאחר חילוץ התכונות)
FIXED_LENGTH = 400 

class FeatureDataset(Dataset):
    """
    מחלקה לניהול וטעינת דאטה-סט של תכונות (Features) שחולצו מראש.
    """
    def __init__(self, mode, split='train'):
        self.mode = mode
        self.split = split
        
        protocol_file = os.path.join(PROTOCOLS_DIR, f"{split}_protocol.txt")
        self.file_list = []
        self.labels = []
        
        if not os.path.exists(protocol_file):
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")

        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    fname = parts[0]
                    label_str = parts[1] 
                    # bonafide (אמיתי) -> 1, spoof (מזויף) -> 0
                    self.file_list.append(fname)
                    self.labels.append(1 if 'bonafide' in label_str else 0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        label = self.labels[idx]
        
        p_wavlm = os.path.join(FEATS_WAVLM, self.split, fname + ".pt")
        p_whisper = os.path.join(FEATS_WHISPER, self.split, fname + ".pt")
        
        try:
            feat = None
            
            if self.mode == 'wavlm':
                # [1, Time, 768] -> [768, Time]
                f = torch.load(p_wavlm).squeeze(0).transpose(0, 1)
                feat = f

            elif self.mode == 'whisper':
                # [1, Time, 768] -> [768, Time]
                f = torch.load(p_whisper).squeeze(0).transpose(0, 1)
                feat = f

            elif self.mode == 'fusion':
                fw = torch.load(p_wavlm).squeeze(0).transpose(0, 1)   # [768, Tw]
                fs = torch.load(p_whisper).squeeze(0).transpose(0, 1) # [768, Ts]
                
                target_len = fw.shape[1]
                
                # אינטרפולציה ל-Whisper
                fs = fs.unsqueeze(0) 
                fs = torch.nn.functional.interpolate(fs, size=target_len, mode='linear', align_corners=False)
                fs = fs.squeeze(0)
                
                # שרשור -> [1536, Time]
                feat = torch.cat((fw, fs), dim=0) 

            # חיתוך/ריפוד לאורך קבוע
            current_len = feat.shape[1]
            if current_len > FIXED_LENGTH:
                feat = feat[:, :FIXED_LENGTH]
            elif current_len < FIXED_LENGTH:
                pad_amt = FIXED_LENGTH - current_len
                feat = torch.nn.functional.pad(feat, (0, pad_amt))
                
            return feat, label

        except Exception as e:
            # במקרה של שגיאה, מחזיר טנסור אפסים
            # print(f"Warning: Error loading file {fname}: {e}") # אפשר להחזיר אם רוצים לראות שגיאות
            dim = 1536 if self.mode == 'fusion' else 768
            return torch.zeros(dim, FIXED_LENGTH), label

def train(args):
    print(f"🚀 Starting Training: MODE={args.mode.upper()} | EPOCHS={args.epochs}")
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    
    train_ds = FeatureDataset(args.mode, split='train')
    # אם הדאטה קטן מאוד, תוריד את batch_size ל-4 או 8
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, drop_last=True)
    
    input_dim = 1536 if args.mode == 'fusion' else 768
    model = Nes2Net(input_channels=input_dim).to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    best_loss = float('inf')

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(feats)
            
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix(loss=loss.item(), acc=100*correct/total)
            
        avg_loss = total_loss / len(train_loader)
        epoch_acc = 100 * correct / total
        print(f"Epoch {epoch+1} Summary: Loss={avg_loss:.4f}, Acc={epoch_acc:.2f}%")
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            save_path = os.path.join(CHECKPOINT_DIR, f"best_model_{args.mode}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Saved Best Model to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['wavlm', 'whisper', 'fusion'])
    parser.add_argument("--epochs", type=int, default=10)
    
    args = parser.parse_args()
    
    train(args)