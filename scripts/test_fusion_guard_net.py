import os
import sys
import csv
import random
import numpy as np
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d

"""
=============================================================================
TEST FusionGuardNet
- loads a specific epoch checkpoint (default: epoch 8)
- evaluates on test split only
- prints loss, accuracy, confusion matrix, precision, recall, f1
- saves every run into a unique timestamped folder so nothing gets overwritten

Expected data structure:
D:\Fusion_Model\dataset\
  data\
    processed\
      test\
        real\*.pt
        fake\*.pt
  checkpoints\
    all_epochs\
      fusion_guard_net_epoch_08.pth
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

DATASET_ROOT = r"D:\Fusion_Model\dataset"

PROCESSED_DIR = os.path.join(DATASET_ROOT, "data", "processed")
CHECKPOINTS_DIR = os.path.join(DATASET_ROOT, "checkpoints")
ALL_EPOCHS_DIR = os.path.join(CHECKPOINTS_DIR, "all_epochs")
RESULTS_DIR = os.path.join(DATASET_ROOT, "results")
TEST_RUNS_DIR = os.path.join(RESULTS_DIR, "test_runs")

TARGET_EPOCH = 8
MODEL_PATH = os.path.join(
    ALL_EPOCHS_DIR,
    f"fusion_guard_net_epoch_{TARGET_EPOCH:02d}.pth"
)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 16
FIXED_LEN = 200
SEED = 42
NUM_WORKERS = 0  # Windows-friendly
PIN_MEMORY = DEVICE == "cuda"

# --------------------------------------------------
# Utils
# --------------------------------------------------

def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dirs():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(TEST_RUNS_DIR, exist_ok=True)

def make_unique_run_dir():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = os.path.join(TEST_RUNS_DIR, f"test_epoch_{TARGET_EPOCH:02d}_{timestamp}")

    suffix = 1
    base_run_dir = run_dir
    while os.path.exists(run_dir):
        run_dir = f"{base_run_dir}_{suffix}"
        suffix += 1

    os.makedirs(run_dir, exist_ok=False)
    return run_dir

def label_to_name(label: int) -> str:
    return "real" if label == 0 else "fake"

# --------------------------------------------------
# Dataset
# --------------------------------------------------

class FusionFeatureDataset(Dataset):
    def __init__(self, split="test", fixed_len=200):
        assert split in {"train", "dev", "test"}

        self.split = split
        self.fixed_len = fixed_len
        self.file_list = []

        class_to_label = {
            "real": 0,
            "fake": 1,
        }

        for cls_name, cls_label in class_to_label.items():
            folder_path = os.path.join(PROCESSED_DIR, split, cls_name)

            if os.path.exists(folder_path):
                for fname in sorted(os.listdir(folder_path)):
                    if fname.endswith(".pt"):
                        self.file_list.append((os.path.join(folder_path, fname), cls_label))
            else:
                print(f"Warning: missing folder {folder_path}")

        self.real_count = sum(1 for _, y in self.file_list if y == 0)
        self.fake_count = sum(1 for _, y in self.file_list if y == 1)

    def __len__(self):
        return len(self.file_list)

    def _fix_length(self, feat):
        # expected [C, T]
        if feat.shape[1] > self.fixed_len:
            feat = feat[:, :self.fixed_len]
        elif feat.shape[1] < self.fixed_len:
            pad_amt = self.fixed_len - feat.shape[1]
            feat = F.pad(feat, (0, pad_amt))
        return feat

    def __getitem__(self, idx):
        file_path, fallback_label = self.file_list[idx]

        try:
            data = torch.load(file_path, map_location="cpu")

            wavlm_feat = data["wavlm"].transpose(0, 1).float()      # [768, T]
            whisper_feat = data["whisper"].transpose(0, 1).float()  # [768, T]
            label = torch.tensor(int(data.get("label", fallback_label)), dtype=torch.long)

            wavlm_feat = self._fix_length(wavlm_feat)
            whisper_feat = self._fix_length(whisper_feat)

            return wavlm_feat, whisper_feat, label, file_path

        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            wavlm_feat = torch.zeros(768, self.fixed_len, dtype=torch.float32)
            whisper_feat = torch.zeros(768, self.fixed_len, dtype=torch.float32)
            label = torch.tensor(fallback_label, dtype=torch.long)
            return wavlm_feat, whisper_feat, label, file_path

# --------------------------------------------------
# Metrics
# --------------------------------------------------

def compute_binary_metrics(y_true, y_pred):
    """
    class 0 = real
    class 1 = fake
    """
    tp = sum((yt == 1 and yp == 1) for yt, yp in zip(y_true, y_pred))
    tn = sum((yt == 0 and yp == 0) for yt, yp in zip(y_true, y_pred))
    fp = sum((yt == 0 and yp == 1) for yt, yp in zip(y_true, y_pred))
    fn = sum((yt == 1 and yp == 0) for yt, yp in zip(y_true, y_pred))

    accuracy = (tp + tn) / max(len(y_true), 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)

    return {
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": accuracy,
        "precision_fake": precision,
        "recall_fake": recall,
        "f1_fake": f1,
    }


def compute_eer(y_true, y_scores):
    """
    computes the Equal Error Rate (EER) and the corresponding threshold from the true labels and predicted scores.
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    
    eer = brentq(lambda x : 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    
    # Find the optimal threshold corresponding to the EER
    optimal_threshold = interp1d(fpr, thresholds)(eer)
    
    return eer, optimal_threshold, fpr, tpr


def save_roc_curve_plot(fpr, tpr, eer, save_path):
    """
    Computes the AUC and saves the ROC curve plot with the EER point marked.
    """
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.plot(eer, 1-eer, 'ro', label=f'EER = {eer:.4f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FAR)')
    plt.ylabel('True Positive Rate (1 - FRR)')
    plt.title(f'ROC Curve - Epoch {TARGET_EPOCH}')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --------------------------------------------------
# Save helpers
# --------------------------------------------------

def save_confusion_matrix_plot(metrics, save_path):
    cm = [
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]],
    ]

    plt.figure(figsize=(5, 4))
    plt.imshow(cm)
    plt.title(f"Confusion Matrix - Epoch {TARGET_EPOCH}")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks([0, 1], ["real", "fake"])
    plt.yticks([0, 1], ["real", "fake"])

    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i][j]), ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_predictions_csv(records, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "true_label",
            "true_label_name",
            "pred_label",
            "pred_label_name",
            "prob_real",
            "prob_fake",
            "is_correct",
        ])

        for r in records:
            writer.writerow([
                r["file"],
                r["true_label"],
                label_to_name(r["true_label"]),
                r["pred_label"],
                label_to_name(r["pred_label"]),
                f"{r['prob_real']:.6f}",
                f"{r['prob_fake']:.6f}",
                int(r["is_correct"]),
            ])

def save_mistakes_csv(mistakes, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "file",
            "true_label",
            "true_label_name",
            "pred_label",
            "pred_label_name",
            "prob_real",
            "prob_fake",
        ])

        for r in mistakes:
            writer.writerow([
                r["file"],
                r["true_label"],
                label_to_name(r["true_label"]),
                r["pred_label"],
                label_to_name(r["pred_label"]),
                f"{r['prob_real']:.6f}",
                f"{r['prob_fake']:.6f}",
            ])

# --------------------------------------------------
# Evaluation
# --------------------------------------------------

def evaluate_test(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    all_labels = []
    all_preds = []
    all_records = []

    with torch.no_grad():
        for wav_feats, whisper_feats, labels, file_paths in loader:
            wav_feats = wav_feats.to(DEVICE, non_blocking=True)
            whisper_feats = whisper_feats.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)

            outputs = model(wav_feats, whisper_feats)
            loss = criterion(outputs, labels)

            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)

            total_loss += loss.item() * labels.size(0)
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            labels_cpu = labels.cpu().tolist()
            preds_cpu = preds.cpu().tolist()
            probs_cpu = probs.cpu().tolist()

            all_labels.extend(labels_cpu)
            all_preds.extend(preds_cpu)

            for fp, yt, yp, pr in zip(file_paths, labels_cpu, preds_cpu, probs_cpu):
                rec = {
                    "file": fp,
                    "true_label": yt,
                    "pred_label": yp,
                    "prob_real": float(pr[0]),
                    "prob_fake": float(pr[1]),
                    "is_correct": (yt == yp),
                }
                all_records.append(rec)

    avg_loss = total_loss / max(total_samples, 1)
    acc = 100.0 * total_correct / max(total_samples, 1)

    metrics = compute_binary_metrics(all_labels, all_preds)
    mistakes = [r for r in all_records if not r["is_correct"]]

    all_scores = [r["prob_fake"] for r in all_records]

    return avg_loss, acc, metrics, all_records, mistakes, all_labels, all_scores

# --------------------------------------------------
# Main test
# --------------------------------------------------

def test():
    print("=" * 60)
    print("🧪 Testing FusionGuardNet")
    print("=" * 60)
    print(f"Using device : {DEVICE}")
    print(f"Processed dir: {PROCESSED_DIR}")
    print(f"Model path   : {MODEL_PATH}")
    print(f"Target epoch : {TARGET_EPOCH}")

    ensure_dirs()
    set_seed(SEED)

    test_ds = FusionFeatureDataset(split="test", fixed_len=FIXED_LEN)
    print(f"Test samples: {len(test_ds)} | real={test_ds.real_count} | fake={test_ds.fake_count}")

    if len(test_ds) == 0:
        print("No test data found.")
        return

    if not os.path.exists(MODEL_PATH):
        print("Epoch checkpoint not found.")
        return

    run_dir = make_unique_run_dir()
    print(f"Run output dir: {run_dir}")

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
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

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc, metrics, all_records, mistakes, y_true, y_scores = evaluate_test(model, test_loader, criterion)

    eer_val, opt_thresh, fpr, tpr = compute_eer(y_true, y_scores)

    roc_png_path = os.path.join(run_dir, f"roc_curve_epoch_{TARGET_EPOCH:02d}.png")

    print("\n" + "=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Epoch tested     : {TARGET_EPOCH}")
    print(f"Test Loss        : {test_loss:.4f}")
    print(f"Test Accuracy    : {test_acc:.2f}%")
    print(f"EER              : {eer_val:.4f} ({eer_val*100:.2f}%)")
    print(f"Optimal Threshold: {opt_thresh:.4f}")
    print()
    print("Confusion Matrix (class 0 = real, class 1 = fake)")
    print(f"TN: {metrics['tn']} | FP: {metrics['fp']}")
    print(f"FN: {metrics['fn']} | TP: {metrics['tp']}")
    print()
    print(f"Precision (fake) : {metrics['precision_fake']:.4f}")
    print(f"Recall    (fake) : {metrics['recall_fake']:.4f}")
    print(f"F1 score  (fake) : {metrics['f1_fake']:.4f}")
    print(f"Total mistakes   : {len(mistakes)}")

    report_path = os.path.join(run_dir, f"fusion_guard_net_test_report_epoch_{TARGET_EPOCH:02d}.txt")
    predictions_csv_path = os.path.join(run_dir, f"all_predictions_epoch_{TARGET_EPOCH:02d}.csv")
    mistakes_csv_path = os.path.join(run_dir, f"mistakes_epoch_{TARGET_EPOCH:02d}.csv")
    cm_png_path = os.path.join(run_dir, f"confusion_matrix_epoch_{TARGET_EPOCH:02d}.png")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("FusionGuardNet Test Report\n")
        f.write("=" * 40 + "\n")
        f.write(f"Run dir: {run_dir}\n")
        f.write(f"Device: {DEVICE}\n")
        f.write(f"Target epoch: {TARGET_EPOCH}\n")
        f.write(f"Checkpoint: {MODEL_PATH}\n")
        f.write(f"Processed dir: {PROCESSED_DIR}\n")
        f.write(f"Test samples: {len(test_ds)}\n")
        f.write(f"Test real/fake: {test_ds.real_count}/{test_ds.fake_count}\n\n")

        f.write(f"Test Loss: {test_loss:.4f}\n")
        f.write(f"Test Accuracy: {test_acc:.2f}%\n\n")

        f.write(f"EER: {eer_val:.4f} ({eer_val*100:.2f}%)\n")
        f.write(f"Optimal Threshold: {opt_thresh:.4f}\n\n")

        f.write("Confusion Matrix (class 0 = real, class 1 = fake)\n")
        f.write(f"TN: {metrics['tn']} | FP: {metrics['fp']}\n")
        f.write(f"FN: {metrics['fn']} | TP: {metrics['tp']}\n\n")

        f.write(f"Precision (fake): {metrics['precision_fake']:.4f}\n")
        f.write(f"Recall    (fake): {metrics['recall_fake']:.4f}\n")
        f.write(f"F1 score  (fake): {metrics['f1_fake']:.4f}\n")
        f.write(f"Total mistakes: {len(mistakes)}\n\n")

        f.write("Saved files\n")
        f.write("-" * 40 + "\n")
        f.write(f"Report: {report_path}\n")
        f.write(f"Predictions CSV: {predictions_csv_path}\n")
        f.write(f"Mistakes CSV: {mistakes_csv_path}\n")
        f.write(f"Confusion matrix PNG: {cm_png_path}\n")

        f.write(f"ROC Curve PNG: {roc_png_path}\n")

    save_predictions_csv(all_records, predictions_csv_path)
    save_mistakes_csv(mistakes, mistakes_csv_path)
    save_confusion_matrix_plot(metrics, cm_png_path)
    save_roc_curve_plot(fpr, tpr, eer_val, roc_png_path)


    print(f"\nTest report saved to: {report_path}")
    print(f"Predictions CSV saved to: {predictions_csv_path}")
    print(f"Mistakes CSV saved to: {mistakes_csv_path}")
    print(f"Confusion matrix saved to: {cm_png_path}")

if __name__ == "__main__":
    test()