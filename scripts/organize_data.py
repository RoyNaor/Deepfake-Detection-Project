import os
import shutil
import random

# --- Path Configurations (Based on your directory structure) ---
SOURCE_ROOT = r"C:\Users\amitn\Downloads\LA" 
PROJECT_ROOT = r"./"

# Quotas for each split (Defining small subsets for the POC)
QUOTAS = {
    'train': {'real': 2000, 'fake': 2000},
    'dev':   {'real': 500,  'fake': 500},
    'test':  {'real': 500,  'fake': 500} # Test will be pulled from the original Evaluation set
}

# Mapping original protocol files and audio directories
DATA_MAP = {
    'train': {
        'proto': os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.train.trn.txt"),
        'audio': os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_train", "flac")
    },
    'dev': {
        'proto': os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.dev.trl.txt"),
        'audio': os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_dev", "flac")
    },
    'test': { 
        'proto': os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_cm_protocols", "ASVspoof2019.LA.cm.eval.trl.txt"),
        'audio': os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_eval", "flac")
    }
}

DEST_BASE_AUDIO = os.path.join(PROJECT_ROOT, "data", "raw_audio")
DEST_PROTO_DIR = os.path.join(PROJECT_ROOT, "data", "protocols")

def process_split(split_name):
    print(f"\n--- Processing split: {split_name.upper()} ---")
    
    # 1. Create destination directory
    dest_dir = os.path.join(DEST_BASE_AUDIO, split_name)
    os.makedirs(dest_dir, exist_ok=True)
    
    # 2. Read original protocol
    proto_path = DATA_MAP[split_name]['proto']
    audio_src_dir = DATA_MAP[split_name]['audio']
    
    real_files = []
    fake_files = []
    
    if not os.path.exists(proto_path):
        print(f"Error: Protocol file not found at {proto_path}")
        return

    with open(proto_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            # LA Format: [Speaker_ID, File_Name, System_ID, Access_Type, Label]
            fname = parts[1]
            label = parts[-1]
            
            if label == 'bonafide':
                real_files.append(fname)
            else:
                fake_files.append(fname)
    
    # 3. Sampling based on defined quotas
    random.shuffle(real_files)
    random.shuffle(fake_files)
    
    selected_real = real_files[:QUOTAS[split_name]['real']]
    selected_fake = fake_files[:QUOTAS[split_name]['fake']]
    
    print(f"Selected: {len(selected_real)} Real samples, {len(selected_fake)} Fake samples.")
    
    # 4. Copying files and generating new protocol
    new_proto_path = os.path.join(DEST_PROTO_DIR, f"{split_name}_protocol.txt")
    all_selected = [(f, 'bonafide') for f in selected_real] + [(f, 'spoof') for f in selected_fake]
    random.shuffle(all_selected)
    
    count = 0
    with open(new_proto_path, 'w') as f_out:
        for fname, label in all_selected:
            src = os.path.join(audio_src_dir, fname + ".flac")
            dst = os.path.join(dest_dir, fname + ".flac")
            
            if os.path.exists(src):
                shutil.copy2(src, dst)
                f_out.write(f"{fname} {label}\n")
                count += 1
            else:
                print(f"Warning: File missing at {src}")
    
    print(f"Success: {count} files copied to {split_name}")

def setup_data():
    print("Initializing Data Setup...")
    os.makedirs(DEST_PROTO_DIR, exist_ok=True)
    for split in ['train', 'dev', 'test']:
        process_split(split)
    print("\n✅ Data Organization Completed Successfully!")

if __name__ == "__main__":
    setup_data()