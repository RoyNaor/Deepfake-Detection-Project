import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import argparse
from tqdm import tqdm
import sys

# נתיבי עבודה בתוך ה-POC
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.nes2net import Nes2Net 

# קבועים
BASE_DIR = parent_dir 
DATA_DIR = os.path.join(BASE_DIR, "data")
PROTOCOLS_DIR = os.path.join(DATA_DIR, "protocols")
FEATS_WAVLM = os.path.join(DATA_DIR, "feats_wavlm")
FEATS_WHISPER = os.path.join(DATA_DIR, "feats_whisper")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FIXED_LENGTH = 400 # תואם ל-4 שניות שחילצנו

class FeatureDataset(Dataset):
    def __init__(self, mode, split='train'):
        self.mode = mode
        self.split = split
        protocol_file = os.path.join(PROTOCOLS_DIR, f"{split}_protocol.txt")
        self.file_list = []
        self.labels = []
        
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    self.file_list.append(parts[0])
                    # bonafide=1, spoof=0
                    self.labels.append(1 if 'bonafide' in parts[1] else 0)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        fname = self.file_list[idx]
        label = self.labels[idx]
        
        # טעינת התכונות שחילצנו בשלב הקודם
        p_wavlm = os.path.join(FEATS_WAVLM, self.split, fname + ".pt")
        p_whisper = os.path.join(FEATS_WHISPER, self.split, fname + ".pt")
        
        try:
            if self.mode == 'wavlm':
                feat = torch.load(p_wavlm).squeeze(0).transpose(0, 1) # [768, Time]
            elif self.mode == 'whisper':
                feat = torch.load(p_whisper).squeeze(0).transpose(0, 1) # [768, Time]
            elif self.mode == 'fusion':
                fw = torch.load(p_wavlm).squeeze(0).transpose(0, 1)    # [768, Tw]
                fs = torch.load(p_whisper).squeeze(0).transpose(0, 1)  # [768, Ts]
                
                # אינטרפולציה מדויקת לסנכרון Whisper עם WavLM
                fs = fs.unsqueeze(0) 
                fs = torch.nn.functional.interpolate(fs, size=fw.shape[1], mode='linear', align_corners=False)
                feat = torch.cat((fw, fs.squeeze(0)), dim=0) # [1536, Time]

            # יישור לאורך קבוע (400 פריימים)
            curr_len = feat.shape[1]
            if curr_len > FIXED_LENGTH:
                feat = feat[:, :FIXED_LENGTH]
            else:
                feat = torch.nn.functional.pad(feat, (0, FIXED_LENGTH - curr_len))
                
            return feat, label
        except:
            dim = 1536 if self.mode == 'fusion' else 768
            return torch.zeros(dim, FIXED_LENGTH), label

def validate(model, loader, criterion):
    model.eval()
    val_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            outputs = model(feats)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return val_loss / len(loader), 100 * correct / total

def train(args):
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    train_loader = DataLoader(FeatureDataset(args.mode, 'train'), batch_size=16, shuffle=True)
    dev_loader = DataLoader(FeatureDataset(args.mode, 'dev'), batch_size=16, shuffle=False)
    
    input_dim = 1536 if args.mode == 'fusion' else 768
    # שימוש ב-ASTP כפי שמוגדר במודל שלך לתוצאות מקסימליות
    model = Nes2Net(input_channels=input_dim, pool_func='ASTP').to(DEVICE)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    best_acc = 0

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for feats, labels in pbar:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(feats), labels)
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

        # שלב התיקוף (Dev) - קריטי ל-Generalization
        val_loss, val_acc = validate(model, dev_loader, criterion)
        print(f"Validation - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(CHECKPOINT_DIR, f"best_{args.mode}.pth"))
            print(f"⭐ New Best Model Saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, required=True, choices=['wavlm', 'whisper', 'fusion'])
    parser.add_argument("--epochs", type=int, default=20)
    train(parser.parse_args())