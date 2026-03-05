import os
import shutil
import random

# -------------------------
# CONFIG
# -------------------------

SOURCE_ROOT = r"C:\Users\amitn\Downloads\LA"
PROJECT_ROOT = r"./"

TOTAL_SAMPLES = 65000
TRAIN_RATIO = 0.8  # 80% train

ORIGINAL_PROTO = os.path.join(
    SOURCE_ROOT,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.train.trn.txt"
)

ORIGINAL_AUDIO = os.path.join(
    SOURCE_ROOT,
    "ASVspoof2019_LA_train",
    "flac"
)

DEST_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "train")
DEST_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "raw", "test")
DEST_PROTO_DIR = os.path.join(PROJECT_ROOT, "data", "protocols")


def setup_data():
    print("🚀 Organizing dataset (65K experiment)")

    os.makedirs(DEST_TRAIN_DIR, exist_ok=True)
    os.makedirs(DEST_TEST_DIR, exist_ok=True)
    os.makedirs(DEST_PROTO_DIR, exist_ok=True)

    real_files = []
    fake_files = []

    print("📖 Reading protocol...")
    with open(ORIGINAL_PROTO, 'r') as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1]
            label = parts[-1]

            if label == "bonafide":
                real_files.append(filename)
            else:
                fake_files.append(filename)

    # Shuffle to ensure randomness, so the model wont learn any order-based patterns
    random.shuffle(real_files)
    random.shuffle(fake_files)

    print(f"Total available: {len(real_files)} real, {len(fake_files)} fake")

    # -------------------------
    # Balanced Selection
    # -------------------------

    max_per_class = TOTAL_SAMPLES // 2

    real_files = real_files[:max_per_class]
    fake_files = fake_files[:max_per_class]

    train_size = int(TRAIN_RATIO * max_per_class)
    test_size = max_per_class - train_size

    train_real = real_files[:train_size]
    test_real = real_files[train_size:]

    train_fake = fake_files[:train_size]
    test_fake = fake_files[train_size:]

    print(f"Train: {len(train_real)} real + {len(train_fake)} fake")
    print(f"Test: {len(test_real)} real + {len(test_fake)} fake")

    # -------------------------
    # Copy Function
    # -------------------------

    # FIX (?): this function dosnt seperate real and fake files into different folders, 
    # it might be causing problems in 'extract_features.py' when it tries to label the data, 
    # because it checks if the file is in the real_set, but all files are in the same folder, 
    # so it might be labeling all files as real, which is a problem. To fix this, we can create two separate folders for real and fake files. 
    
    def copy_subset(file_list, dest_dir, proto_name, real_set):
        proto_path = os.path.join(DEST_PROTO_DIR, proto_name)

        with open(proto_path, "w") as f_proto:
            for fname in file_list:
                src = os.path.join(ORIGINAL_AUDIO, fname + ".flac")
                dst = os.path.join(dest_dir, fname + ".flac")

                try:
                    shutil.copy2(src, dst)

                    label_str = "bonafide" if fname in real_set else "spoof"
                    f_proto.write(f"{fname} {label_str}\n")

                except FileNotFoundError:
                    print(f"⚠️ Missing file: {fname}")

    real_set = set(real_files)

    # Train
    train_all = train_real + train_fake
    random.shuffle(train_all)
    copy_subset(train_all, DEST_TRAIN_DIR, "train_protocol.txt", real_set)

    # Test
    test_all = test_real + test_fake
    random.shuffle(test_all)
    copy_subset(test_all, DEST_TEST_DIR, "test_protocol.txt", real_set)

    print("✅ Done. 65K dataset ready.")


if __name__ == "__main__":
    setup_data()
