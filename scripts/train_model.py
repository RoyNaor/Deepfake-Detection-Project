import os
import sys
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

"""
=============================================================================
TRAIN FusionGuardNet
- trains only the fusion model on pre-extracted frozen features
- uses:
    train -> optimization
    dev   -> model selection
    test  -> final evaluation
Expected data structure:
data/
  processed/
    train/
      real/*.pt
      fake/*.pt
    dev/
      real/*.pt
      fake/*.pt
    test/
      real/*.pt
      fake/*.pt
=============================================================================
"""

# --------------------------------------------------
# Path setup
# --------------------------------------------------

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.fusion_guard_net import FusionGuardNet

# --------------------------------------------------
# Config
# --------------------------------------------------

DATA_ROOT = os.path.join(parent_dir, "data")
PROCESSED_DIR = os.path.join(DATA_ROOT, "processed")

RESULTS_DIR = os.path.join(parent_dir, "results")
CHECKPOINTS_DIR = os.path.join(parent_dir, "checkpoints")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
EPOCHS = 15
LR = 1e-4
WEIGHT_DECAY = 1e-5
FIXED_LEN = 200
SEED = 42
NUM_WORKERS = 0  # Windows-friendly

# --------------------------------------------------
# Utils
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)

# --------------------------------------------------
# Dataset
# --------------------------------------------------

class FusionFeatureDataset(Dataset):
    def __init__(self, split="train", fixed_len=200):
        assert split in {"train", "dev", "test"}

        self.split = split
        self.fixed_len = fixed_len
        self.file_list = []

        for cls_name in ["real", "fake"]:
            folder_path = os.path.join(PROCESSED_DIR, split, cls_name)

            if os.path.exists(folder_path):
                for fname in sorted(os.listdir(folder_path)):
                    if fname.endswith(".pt"):
                        self.file_list.append(os.path.join(folder_path, fname))
            else:
                print(f"Warning: missing folder {folder_path}")

    def __len__(self):
        return len(self.file_list)

    def _fix_length(self, feat):
        # feat expected [C, T]
        if feat.shape[1] > self.fixed_len:
            feat = feat[:, :self.fixed_len]
        elif feat.shape[1] < self.fixed_len:
            pad_amt = self.fixed_len - feat.shape[1]
            feat = F.pad(feat, (0, pad_amt))
        return feat

    def __getitem__(self, idx):
        file_path = self.file_list[idx]

        try:
            data = torch.load(file_path, map_location="cpu")

            wavlm_feat = data["wavlm"].transpose(0, 1).float()      # [768, T]
            whisper_feat = data["whisper"].transpose(0, 1).float()  # [768, T]
            label = torch.tensor(int(data["label"]), dtype=torch.long)

            wavlm_feat = self._fix_length(wavlm_feat)
            whisper_feat = self._fix_length(whisper_feat)

            return wavlm_feat, whisper_feat, label

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            wavlm_feat = torch.zeros(768, self.fixed_len, dtype=torch.float32)
            whisper_feat = torch.zeros(768, self.fixed_len, dtype=torch.float32)
            label = torch.tensor(0, dtype=torch.long)
            return wavlm_feat, whisper_feat, label

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for wav_feats, whisper_feats, labels in loader:
            wav_feats = wav_feats.to(DEVICE)
            whisper_feats = whisper_feats.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(wav_feats, whisper_feats)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

    if total_samples == 0:
        return 0.0, 0.0

    avg_loss = total_loss / total_samples
    acc = 100.0 * total_correct / total_samples
    return avg_loss, acc

# --------------------------------------------------
# Training
# --------------------------------------------------

def train():
    print("=" * 60)
    print("🚀 Training FusionGuardNet")
    print("=" * 60)
    print(f"Using device: {DEVICE}")
    print(f"Processed dir: {PROCESSED_DIR}")

    ensure_dirs()
    set_seed(SEED)

    train_ds = FusionFeatureDataset(split="train", fixed_len=FIXED_LEN)
    dev_ds = FusionFeatureDataset(split="dev", fixed_len=FIXED_LEN)
    test_ds = FusionFeatureDataset(split="test", fixed_len=FIXED_LEN)

    print(f"Train samples: {len(train_ds)}")
    print(f"Dev samples:   {len(dev_ds)}")
    print(f"Test samples:  {len(test_ds)}")

    if len(train_ds) == 0 or len(dev_ds) == 0 or len(test_ds) == 0:
        print("❌ Missing processed data. Run extract_features.py first.")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=(DEVICE == "cuda")
    )

    model = FusionGuardNet(
        feature_channels=768,
        nes_ratio=(8, 8),
        dilation=2,
        pool_func="mean",   # אפשר גם "ASTP" אם תרצה
        se_ratio=(8,),
        num_classes=2,
        use_softmax_gate=True
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    best_dev_acc = 0.0
    best_epoch = 0

    best_model_path = os.path.join(CHECKPOINTS_DIR, "best_fusion_guard_net.pth")
    last_model_path = os.path.join(CHECKPOINTS_DIR, "last_fusion_guard_net.pth")

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for wav_feats, whisper_feats, labels in train_loader:
            wav_feats = wav_feats.to(DEVICE)
            whisper_feats = whisper_feats.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(wav_feats, whisper_feats)
            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total if running_total > 0 else 0.0
        train_acc = 100.0 * running_correct / running_total if running_total > 0 else 0.0

        dev_loss, dev_acc = evaluate(model, dev_loader, criterion)

        # save last checkpoint every epoch
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_dev_acc": best_dev_acc,
            },
            last_model_path
        )

        # save best checkpoint
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dev_acc": best_dev_acc,
                },
                best_model_path
            )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev Acc: {dev_acc:.2f}%"
        )

    print("\n✅ Training finished.")
    print(f"🏆 Best dev acc: {best_dev_acc:.2f}% at epoch {best_epoch}")

    # --------------------------------------------------
    # Final test with best checkpoint
    # --------------------------------------------------

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("⚠️ Best checkpoint not found, evaluating current model.")

    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"🧪 Final Test Loss: {test_loss:.4f}")
    print(f"🧪 Final Test Acc : {test_acc:.2f}%")

    # save final summary
    summary_path = os.path.join(RESULTS_DIR, "fusion_guard_net_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FusionGuardNet Training Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Train samples: {len(train_ds)}\n")
        f.write(f"Dev samples: {len(dev_ds)}\n")
        f.write(f"Test samples: {len(test_ds)}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LR}\n")
        f.write(f"Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"Best dev acc: {best_dev_acc:.2f}% (epoch {best_epoch})\n")
        f.write(f"Final test loss: {test_loss:.4f}\n")
        f.write(f"Final test acc: {test_acc:.2f}%\n\n")

        f.write("Epoch History\n")
        f.write("-" * 40 + "\n")
        for row in history:
            f.write(
                f"Epoch {row['epoch']:02d} | "
                f"Train Loss: {row['train_loss']:.4f} | "
                f"Train Acc: {row['train_acc']:.2f}% | "
                f"Dev Loss: {row['dev_loss']:.4f} | "
                f"Dev Acc: {row['dev_acc']:.2f}%\n"
            )

    print(f"📝 Summary saved to: {summary_path}")
    print(f"💾 Best checkpoint:  {best_model_path}")
    print(f"💾 Last checkpoint:  {last_model_path}")

if __name__ == "__main__":
    train()