import os
import sys
import random
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==================================================
# TRAIN FusionGuardNet
# - trains only the fusion model on pre-extracted frozen features
# - uses:
#     train -> optimization
#     dev   -> model selection
#     test  -> final evaluation
#
# Expected data structure:
# D:\Fusion_Model\dataset\data\processed\
#   train\real\*.pt
#   train\fake\*.pt
#   dev\real\*.pt
#   dev\fake\*.pt
#   test\real\*.pt
#   test\fake\*.pt
# ==================================================

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

DATASET_ROOT = r"D:\Fusion_Model\dataset"

PROCESSED_DIR = os.path.join(DATASET_ROOT, "data", "processed")
RESULTS_DIR = os.path.join(DATASET_ROOT, "results")
CHECKPOINTS_DIR = os.path.join(DATASET_ROOT, "checkpoints")
ALL_EPOCHS_DIR = os.path.join(CHECKPOINTS_DIR, "all_epochs")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
EPOCHS = 8
LR = 1e-4
WEIGHT_DECAY = 1e-4
FIXED_LEN = 200
SEED = 42

NUM_WORKERS = 2
PIN_MEMORY = DEVICE == "cuda"

GRAD_CLIP_NORM = 1.0
EARLY_STOPPING_PATIENCE = 4

# --------------------------------------------------
# Utils
# --------------------------------------------------

def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(ALL_EPOCHS_DIR, exist_ok=True)


def count_labels(file_list: List[Tuple[str, int]]) -> Tuple[int, int]:
    real_count = sum(1 for _, label in file_list if label == 0)
    fake_count = sum(1 for _, label in file_list if label == 1)
    return real_count, fake_count


def save_training_plots(history, results_dir):
    epochs = [row["epoch"] for row in history]

    train_losses = [row["train_loss"] for row in history]
    dev_losses = [row["dev_loss"] for row in history]

    train_accs = [row["train_acc"] for row in history]
    dev_accs = [row["dev_acc"] for row in history]

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, dev_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_accs, label="Train Accuracy")
    plt.plot(epochs, dev_accs, label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("Training vs Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, "accuracy_curve.png"))
    plt.close()


# --------------------------------------------------
# Dataset
# --------------------------------------------------

class FusionFeatureDataset(Dataset):
    def __init__(self, split: str = "train", fixed_len: int = 200):
        assert split in {"train", "dev", "test"}

        self.split = split
        self.fixed_len = fixed_len
        self.file_list: List[Tuple[str, int]] = []

        class_to_label = {
            "real": 0,
            "fake": 1,
        }

        for cls_name, cls_label in class_to_label.items():
            folder_path = os.path.join(PROCESSED_DIR, split, cls_name)

            if not os.path.exists(folder_path):
                print(f"Warning: missing folder {folder_path}")
                continue

            for fname in sorted(os.listdir(folder_path)):
                if fname.endswith(".pt"):
                    self.file_list.append((os.path.join(folder_path, fname), cls_label))

        self.real_count, self.fake_count = count_labels(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def _fix_length(self, feat: torch.Tensor) -> torch.Tensor:
        if feat.shape[1] > self.fixed_len:
            feat = feat[:, :self.fixed_len]
        elif feat.shape[1] < self.fixed_len:
            pad_amt = self.fixed_len - feat.shape[1]
            feat = F.pad(feat, (0, pad_amt))
        return feat

    def __getitem__(self, idx: int):
        file_path, fallback_label = self.file_list[idx]

        try:
            data = torch.load(file_path, map_location="cpu")

            wavlm_feat = data["wavlm"].transpose(0, 1).float()
            whisper_feat = data["whisper"].transpose(0, 1).float()

            label_value = int(data.get("label", fallback_label))
            label = torch.tensor(label_value, dtype=torch.long)

            wavlm_feat = self._fix_length(wavlm_feat)
            whisper_feat = self._fix_length(whisper_feat)

            return wavlm_feat, whisper_feat, label

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            wavlm_feat = torch.zeros(768, self.fixed_len, dtype=torch.float32)
            whisper_feat = torch.zeros(768, self.fixed_len, dtype=torch.float32)
            label = torch.tensor(fallback_label, dtype=torch.long)
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
            wav_feats = wav_feats.to(DEVICE, non_blocking=True)
            whisper_feats = whisper_feats.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

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
    print(f"Results dir:   {RESULTS_DIR}")
    print(f"Checkpoints:   {CHECKPOINTS_DIR}")
    print(f"All epochs:    {ALL_EPOCHS_DIR}")

    ensure_dirs()
    set_seed(SEED)

    train_ds = FusionFeatureDataset(split="train", fixed_len=FIXED_LEN)
    dev_ds = FusionFeatureDataset(split="dev", fixed_len=FIXED_LEN)
    test_ds = FusionFeatureDataset(split="test", fixed_len=FIXED_LEN)

    print("\nDataset stats:")
    print(f"Train samples: {len(train_ds)} | real={train_ds.real_count} | fake={train_ds.fake_count}")
    print(f"Dev samples:   {len(dev_ds)} | real={dev_ds.real_count} | fake={dev_ds.fake_count}")
    print(f"Test samples:  {len(test_ds)} | real={test_ds.real_count} | fake={test_ds.fake_count}")

    if len(train_ds) == 0 or len(dev_ds) == 0 or len(test_ds) == 0:
        print("Missing processed data. Run extract_features.py first.")
        return

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    dev_loader = DataLoader(
        dev_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=(NUM_WORKERS > 0),
    )

    model = FusionGuardNet(
        feature_channels=768,
        nes_ratio=(8, 8),
        dilation=2,
        pool_func="mean",
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

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    best_dev_acc = 0.0
    best_dev_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0

    best_model_path = os.path.join(CHECKPOINTS_DIR, "best_fusion_guard_net.pth")
    last_model_path = os.path.join(CHECKPOINTS_DIR, "last_fusion_guard_net.pth")

    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for wav_feats, whisper_feats, labels in train_loader:
            wav_feats = wav_feats.to(DEVICE, non_blocking=True)
            whisper_feats = whisper_feats.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(wav_feats, whisper_feats)
            loss = criterion(outputs, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
            optimizer.step()

            running_loss += loss.item() * labels.size(0)

            preds = outputs.argmax(dim=1)
            running_correct += (preds == labels).sum().item()
            running_total += labels.size(0)

        train_loss = running_loss / running_total if running_total > 0 else 0.0
        train_acc = 100.0 * running_correct / running_total if running_total > 0 else 0.0

        dev_loss, dev_acc = evaluate(model, dev_loader, criterion)
        scheduler.step(dev_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        epoch_checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_loss,
            "train_acc": train_acc,
            "dev_loss": dev_loss,
            "dev_acc": dev_acc,
            "best_dev_acc": best_dev_acc,
            "best_dev_loss": best_dev_loss,
            "current_lr": current_lr,
        }

        epoch_model_path = os.path.join(
            ALL_EPOCHS_DIR,
            f"fusion_guard_net_epoch_{epoch:02d}.pth"
        )
        torch.save(epoch_checkpoint, epoch_model_path)

        torch.save(epoch_checkpoint, last_model_path)

        improved = False
        if dev_acc > best_dev_acc:
            best_dev_acc = dev_acc
            best_epoch = epoch
            improved = True

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            improved = True

        if improved:
            epochs_without_improvement = 0
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "train_acc": train_acc,
                    "dev_loss": dev_loss,
                    "dev_acc": dev_acc,
                    "best_dev_acc": best_dev_acc,
                    "best_dev_loss": best_dev_loss,
                    "current_lr": current_lr,
                },
                best_model_path
            )
        else:
            epochs_without_improvement += 1

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "dev_loss": dev_loss,
                "dev_acc": dev_acc,
                "lr": current_lr,
            }
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Dev Loss: {dev_loss:.4f} | "
            f"Dev Acc: {dev_acc:.2f}% | "
            f"LR: {current_lr:.6f}"
        )
        print(f"💾 Saved epoch checkpoint: {epoch_model_path}")

        if len(history) >= 2:
            prev = history[-2]
            if train_loss < prev["train_loss"] and dev_loss > prev["dev_loss"]:
                print("Possible overfitting: train loss decreased while validation loss increased.")

        if epochs_without_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print("\nTraining finished.")
    print(f"Best dev acc: {best_dev_acc:.2f}% at epoch {best_epoch}")
    print(f"Best dev loss: {best_dev_loss:.4f}")

    if os.path.exists(best_model_path):
        checkpoint = torch.load(best_model_path, map_location=DEVICE)
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Best checkpoint not found, evaluating current model.")

    test_loss, test_acc = evaluate(model, test_loader, criterion)

    print(f"Final Test Loss: {test_loss:.4f}")
    print(f"Final Test Acc : {test_acc:.2f}%")

    summary_path = os.path.join(RESULTS_DIR, "fusion_guard_net_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("FusionGuardNet Training Summary\n")
        f.write("=" * 40 + "\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Processed dir: {PROCESSED_DIR}\n")
        f.write(f"Train samples: {len(train_ds)}\n")
        f.write(f"Dev samples: {len(dev_ds)}\n")
        f.write(f"Test samples: {len(test_ds)}\n")
        f.write(f"Train real/fake: {train_ds.real_count}/{train_ds.fake_count}\n")
        f.write(f"Dev real/fake: {dev_ds.real_count}/{dev_ds.fake_count}\n")
        f.write(f"Test real/fake: {test_ds.real_count}/{test_ds.fake_count}\n")
        f.write(f"Epochs requested: {EPOCHS}\n")
        f.write(f"Epochs completed: {len(history)}\n")
        f.write(f"Batch size: {BATCH_SIZE}\n")
        f.write(f"Learning rate: {LR}\n")
        f.write(f"Weight decay: {WEIGHT_DECAY}\n")
        f.write(f"Fixed len: {FIXED_LEN}\n")
        f.write(f"Workers: {NUM_WORKERS}\n")
        f.write(f"Gradient clipping norm: {GRAD_CLIP_NORM}\n")
        f.write(f"Early stopping patience: {EARLY_STOPPING_PATIENCE}\n")
        f.write("LR scheduler: ReduceLROnPlateau(factor=0.5, patience=2)\n")
        f.write(f"Best dev acc: {best_dev_acc:.2f}% (epoch {best_epoch})\n")
        f.write(f"Best dev loss: {best_dev_loss:.4f}\n")
        f.write(f"Final test loss: {test_loss:.4f}\n")
        f.write(f"Final test acc: {test_acc:.2f}%\n")
        f.write(f"All epoch checkpoints dir: {ALL_EPOCHS_DIR}\n\n")

        f.write("Epoch History\n")
        f.write("-" * 40 + "\n")
        for row in history:
            f.write(
                f"Epoch {row['epoch']:02d} | "
                f"Train Loss: {row['train_loss']:.4f} | "
                f"Train Acc: {row['train_acc']:.2f}% | "
                f"Dev Loss: {row['dev_loss']:.4f} | "
                f"Dev Acc: {row['dev_acc']:.2f}% | "
                f"LR: {row['lr']:.6f}\n"
            )

    save_training_plots(history, RESULTS_DIR)

    print(f"Summary saved to: {summary_path}")
    print(f"Loss/accuracy plots saved to: {RESULTS_DIR}")
    print(f"Best checkpoint:  {best_model_path}")
    print(f"Last checkpoint:  {last_model_path}")
    print(f"All epoch checkpoints dir: {ALL_EPOCHS_DIR}")


if __name__ == "__main__":
    train()