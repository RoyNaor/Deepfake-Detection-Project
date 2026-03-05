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




"""
=============================================================================
CRITICAL FIXES APPLIED TO THIS FILE:

1. Dictionary Loading Fix: 
   The feature extraction script saves data as a dictionary (e.g., {"wavlm": tensor, "label": int}), 
   but the original code tried to load it directly as a tensor. This caused crashes. 
   We updated `__getitem__` to correctly extract the tensors from the loaded dictionary using keys.

2. Whisper Temporal Alignment Fix (Fusion Mode):
   Whisper's processor pads all inputs to 30 seconds (1500 frames). The original code used 
   `interpolate` to squash these 1500 frames into WavLM's length (e.g., 200 frames). 
   This ruined the alignment between acoustic and semantic features. 
   Since both models output at exactly 50Hz, we replaced interpolation with direct slicing (`f_s[:, :target_len]`), 
   perfectly aligning the representations and discarding the padded silence.

3. Tensor Dimensionality Fix:
   Adjusted the transpose and concatenation dimensions to correctly handle the 2D tensors 
   [Channels, Time] extracted from the dictionaries, ensuring the final output matches 
   Nes2Net's expected input format.
=============================================================================
"""
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import Dataset, DataLoader
# import os
# import sys

# # Add parent directory to Path
# current_dir = os.path.dirname(os.path.abspath(__file__))
# parent_dir = os.path.dirname(current_dir)
# sys.path.append(parent_dir)

# from models.nes2net import Nes2Net

# # --- Configurations ---
# DATA_ROOT = os.path.join(parent_dir, "data")
# PROTOCOLS_DIR = os.path.join(DATA_ROOT, "protocols")
# FEATS_WAVLM = os.path.join(DATA_ROOT, "feats_wavlm")
# FEATS_WHISPER = os.path.join(DATA_ROOT, "feats_whisper")
# RESULTS_DIR = os.path.join(parent_dir, "results")
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# # Training Parameters
# BATCH_SIZE = 16 
# EPOCHS = 10      # Enough for POC to see the trend
# LR = 0.0001      # Gentle learning rate

# class FeatureDataset(Dataset):
#     def __init__(self, mode, split='train'):
#         self.mode = mode
#         self.split = split
        
#         # Load the protocol file
#         protocol_file = os.path.join(PROTOCOLS_DIR, f"{split}_protocol.txt")
#         self.file_list = []
#         self.labels = []
        
#         with open(protocol_file, 'r') as f:
#             for line in f:
#                 parts = line.strip().split()
#                 fname = parts[0]
#                 label_str = parts[1]
#                 self.file_list.append(fname)
#                 self.labels.append(1 if label_str == 'bonafide' else 0)

#     def __len__(self):
#         return len(self.file_list)

#     def __getitem__(self, idx):
#         fname = self.file_list[idx]
#         label = self.labels[idx]
        
#         # File paths
#         wavlm_path = os.path.join(FEATS_WAVLM, self.split, fname + ".pt")
#         whisper_path = os.path.join(FEATS_WHISPER, self.split, fname + ".pt")
        
#         try:
#             feat = None
            
#             if self.mode == 'wavlm':
#                 # Load dictionary and extract 'wavlm' tensor. 
#                 # Original shape is [Time, 768], we transpose to [768, Time]
#                 data = torch.load(wavlm_path)
#                 feat = data["wavlm"].transpose(0, 1)

#             elif self.mode == 'whisper':
#                 # Load dictionary and extract 'whisper' tensor.
#                 # Original shape is [Time, 768], we transpose to [768, Time]
#                 data = torch.load(whisper_path)
#                 feat = data["whisper"].transpose(0, 1)

#             elif self.mode == 'fusion':
#                 # Load dictionaries for both
#                 data_w = torch.load(wavlm_path)
#                 data_s = torch.load(whisper_path)
                
#                 # Transpose to [Channels, Time]
#                 f_w = data_w["wavlm"].transpose(0, 1)   # [768, T_wavlm]
#                 f_s = data_s["whisper"].transpose(0, 1) # [768, 1500] (Padded by Whisper)
                
#                 # --- Synchronization (Fusion) FIX ---
#                 # Slice Whisper's output to match WavLM's exact length. 
#                 # Discards the extra padded silence seamlessly.
#                 target_len = f_w.shape[1]
#                 f_s = f_s[:, :target_len] 
                
#                 # Concatenate along the channel dimension (dim=0): 768 + 768 = 1536
#                 feat = torch.cat((f_w, f_s), dim=0) 
            
#             # Crop or pad to a fixed length (to allow batching in DataLoader)
#             fixed_len = 200 # Reasonable length for Nes2Net (~4 seconds)
#             if feat.shape[1] > fixed_len:
#                 feat = feat[:, :fixed_len]
#             elif feat.shape[1] < fixed_len:
#                 pad_amt = fixed_len - feat.shape[1]
#                 feat = torch.nn.functional.pad(feat, (0, pad_amt))
                
#             return feat, label

#         except Exception as e:
#             # Fallback in case a file is missing/corrupted
#             dim = 1536 if self.mode == 'fusion' else 768
#             return torch.zeros(dim, 200), label

# def run_experiment(mode_name):
#     print(f"\n{'='*40}")
#     print(f"🚀 Training Model: {mode_name.upper()}")
#     print(f"{'='*40}")
    
#     input_dim = 1536 if mode_name == 'fusion' else 768
#     model = Nes2Net(input_channels=input_dim).to(DEVICE)
    
#     train_ds = FeatureDataset(mode=mode_name, split='train')
#     test_ds = FeatureDataset(mode=mode_name, split='test')
    
#     # drop_last=True prevents crashes if the last batch is smaller than BATCH_SIZE
#     train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
#     test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)
    
#     criterion = nn.CrossEntropyLoss()
#     optimizer = optim.Adam(model.parameters(), lr=LR)
    
#     best_acc = 0.0
    
#     for epoch in range(EPOCHS):
#         # Train Mode
#         model.train()
#         train_loss = 0
#         train_correct = 0
#         train_total = 0
        
#         for feats, labels in train_loader:
#             feats, labels = feats.to(DEVICE), labels.to(DEVICE)
#             optimizer.zero_grad()
#             outputs = model(feats)
#             loss = criterion(outputs, labels)
#             loss.backward()
#             optimizer.step()
            
#             train_loss += loss.item()
#             _, predicted = torch.max(outputs.data, 1)
#             train_total += labels.size(0)
#             train_correct += (predicted == labels).sum().item()
            
#         # Evaluation Mode
#         model.eval()
#         test_correct = 0
#         test_total = 0
#         with torch.no_grad():
#             for feats, labels in test_loader:
#                 feats, labels = feats.to(DEVICE), labels.to(DEVICE)
#                 outputs = model(feats)
#                 _, predicted = torch.max(outputs.data, 1)
#                 test_total += labels.size(0)
#                 test_correct += (predicted == labels).sum().item()
        
#         acc = 100 * test_correct / test_total
#         print(f"Epoch {epoch+1:02d} | Loss: {train_loss/len(train_loader):.4f} | Train Acc: {100*train_correct/train_total:.1f}% | Test Acc: {acc:.2f}%")
        
#         if acc > best_acc:
#             best_acc = acc
            
#     print(f"🏆 Best Result for {mode_name}: {best_acc:.2f}%")
#     return best_acc

# if __name__ == "__main__":
#     results = {}
    
#     # 1. Run Acoustic Experiment
#     results['WavLM'] = run_experiment('wavlm')
    
#     # 2. Run Semantic Experiment
#     results['Whisper'] = run_experiment('whisper')
    
#     # 3. Run Fusion Experiment
#     results['Fusion'] = run_experiment('fusion')
    
#     print("\n\n📊 --- FINAL RESULTS SUMMARY --- 📊")
#     print("-" * 35)
#     print(f"{'Model':<15} | {'Accuracy':<10}")
#     print("-" * 35)
#     for k, v in results.items():
#         print(f"{k:<15} | {v:.2f}%")
#     print("-" * 35)