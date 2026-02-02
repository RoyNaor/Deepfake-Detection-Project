import torch
from torch.utils.data import DataLoader
import os
import sys

# נתיבים
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from models.nes2net import Nes2Net
from scripts.train_features import FeatureDataset

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_DIR = os.path.join(parent_dir, "checkpoints")

def evaluate(mode, checkpoint_path):
    print(f"Testing Mode: {mode.upper()}...")
    
    # טעינת דאטה (מתיקיית ה-test האמיתית)
    dataset = FeatureDataset(mode=mode, split='test')
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    # הגדרת ממד קלט
    input_dim = 1536 if mode == 'fusion' else 768
    model = Nes2Net(input_channels=input_dim, pool_func='ASTP').to(DEVICE)
    
    # טעינת המשקולות
    if not os.path.exists(checkpoint_path):
        print(f"Skipping {mode}: Checkpoint not found at {checkpoint_path}")
        return None
        
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))
    model.eval()
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for feats, labels in loader:
            feats, labels = feats.to(DEVICE), labels.to(DEVICE)
            outputs = model(feats)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
    accuracy = 100 * correct / total
    return accuracy

if __name__ == "__main__":
    results = {}
    
    # רשימת המודלים לבדיקה
    experiments = [
        ('wavlm', f"best_wavlm.pth"),
        ('whisper', f"best_whisper.pth"),
        ('fusion', f"best_fusion.pth")
    ]
    
    for mode, ckpt in experiments:
        ckpt_path = os.path.join(CHECKPOINT_DIR, ckpt)
        acc = evaluate(mode, ckpt_path)
        if acc is not None:
            results[mode] = acc
            
    print("\n" + "="*40)
    print(f"{'MODEL MODE':<15} | {'TEST ACCURACY':<15}")
    print("-" * 40)
    for mode, acc in results.items():
        print(f"{mode.upper():<15} | {acc:>13.2f}%")
    print("="*40)