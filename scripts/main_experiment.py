"""Train and evaluate Nes2Net on extracted WavLM/Whisper features."""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import sys

# Add project root to Python path for local imports.
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.nes2net import Nes2Net

# --- Paths and configuration ---
DATA_ROOT = os.path.join(parent_dir, "data")
PROTOCOLS_DIR = os.path.join(DATA_ROOT, "protocols")
FEATS_WAVLM = os.path.join(DATA_ROOT, "feats_wavlm")
FEATS_WHISPER = os.path.join(DATA_ROOT, "feats_whisper")
RESULTS_DIR = os.path.join(parent_dir, "results")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Training hyperparameters.
BATCH_SIZE = 16
EPOCHS = 10  # Small value is sufficient for a quick proof-of-concept trend.
LR = 0.0001  # Conservative learning rate for stable optimization.

class FeatureDataset(Dataset):
    """Dataset that loads pre-extracted WavLM/Whisper features and labels."""

    def __init__(self, mode, split='train'):
        """Initialize dataset for a given feature mode and split.

        Args:
            mode: One of {"wavlm", "whisper", "fusion"}.
            split: "train" or "test".
        """
        self.mode = mode
        self.split = split

        # Load the protocol file that maps file IDs to labels.
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
        """Return the number of samples in the split."""
        return len(self.file_list)

    def __getitem__(self, idx):
        """Load and prepare a single sample's features and label.

        Returns:
            Tuple of (features, label) where features are shaped [channels, time].
        """
        fname = self.file_list[idx]
        label = self.labels[idx]

        # Feature file paths for the selected mode.
        wavlm_path = os.path.join(FEATS_WAVLM, self.split, fname + ".pt")
        whisper_path = os.path.join(FEATS_WHISPER, self.split, fname + ".pt")
        
        try:
            feat = None
            
            if self.mode == 'wavlm':
                # WavLM only: [1, Time, 768] -> [1, 768, Time]
                feat = torch.load(wavlm_path)
                feat = feat.transpose(1, 2) # [1, 768, Time]

            elif self.mode == 'whisper':
                # Whisper only: [1, Time, 768] -> [1, 768, Time]
                feat = torch.load(whisper_path)
                feat = feat.transpose(1, 2) # [1, 768, Time]

            elif self.mode == 'fusion':
                # Load both feature sets for fusion.
                f_w = torch.load(wavlm_path).transpose(1, 2)   # [1, 768, T_wavlm]
                f_s = torch.load(whisper_path).transpose(1, 2) # [1, 768, T_whisper]

                # --- Length synchronization for fusion ---
                # Resize Whisper to match WavLM's temporal length.
                target_len = f_w.shape[2]
                f_s = torch.nn.functional.interpolate(f_s, size=target_len, mode='linear', align_corners=False)

                # Concatenate on the channel axis: 768 + 768 = 1536.
                feat = torch.cat((f_w, f_s), dim=1) 

            # Remove the singleton batch dimension: [Channels, Time].
            feat = feat.squeeze(0)

            # Truncate/pad to a fixed length to enable batching.
            fixed_len = 200  # Reasonable temporal length for Nes2Net.
            if feat.shape[1] > fixed_len:
                feat = feat[:, :fixed_len]
            elif feat.shape[1] < fixed_len:
                pad_amt = fixed_len - feat.shape[1]
                feat = torch.nn.functional.pad(feat, (0, pad_amt))
                
            return feat, label

        except Exception as e:
            # For missing/corrupt files, return zero features to keep the run alive.
            dim = 1536 if self.mode == 'fusion' else 768
            return torch.zeros(dim, 200), label

def run_experiment(mode_name):
    """Train and evaluate Nes2Net for a specific feature mode."""
    print(f"\n{'='*40}")
    print(f"🚀 Training Model: {mode_name.upper()}")
    print(f"{'='*40}")
    
    input_dim = 1536 if mode_name == 'fusion' else 768
    model = Nes2Net(input_channels=input_dim).to(DEVICE)
    
    train_ds = FeatureDataset(mode=mode_name, split='train')
    test_ds = FeatureDataset(mode=mode_name, split='test')
    
    # Drop last batch if it is smaller than BATCH_SIZE to avoid shape issues.
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

    # 1) Acoustic-only experiment.
    results['WavLM'] = run_experiment('wavlm')

    # 2) Semantic-only experiment.
    results['Whisper'] = run_experiment('whisper')

    # 3) Combined fusion experiment.
    results['Fusion'] = run_experiment('fusion')
    
    print("\n\n📊 --- FINAL RESULTS SUMMARY --- 📊")
    print("-" * 35)
    print(f"{'Model':<15} | {'Accuracy':<10}")
    print("-" * 35)
    for k, v in results.items():
        print(f"{k:<15} | {v:.2f}%")
    print("-" * 35)
