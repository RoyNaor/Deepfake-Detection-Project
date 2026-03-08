import os
import shutil
import random
from tqdm import tqdm

# --------------------------------------------------
# Paths
# --------------------------------------------------

SOURCE_ROOT = r"C:\Users\amitn\Downloads\LA"
PROJECT_ROOT = "./"

DEST_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DEST_PROTO = os.path.join(PROJECT_ROOT, "data", "protocols")

# --------------------------------------------------
# Dataset map (official ASVspoof splits)
# --------------------------------------------------

DATA_MAP = {
    "train": {
        "proto": os.path.join(
            SOURCE_ROOT,
            "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.train.trn.txt"
        ),
        "audio": os.path.join(
            SOURCE_ROOT,
            "ASVspoof2019_LA_train",
            "flac"
        )
    },

    "dev": {
        "proto": os.path.join(
            SOURCE_ROOT,
            "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.dev.trl.txt"
        ),
        "audio": os.path.join(
            SOURCE_ROOT,
            "ASVspoof2019_LA_dev",
            "flac"
        )
    },

    "test": {
        "proto": os.path.join(
            SOURCE_ROOT,
            "ASVspoof2019_LA_cm_protocols",
            "ASVspoof2019.LA.cm.eval.trl.txt"
        ),
        "audio": os.path.join(
            SOURCE_ROOT,
            "ASVspoof2019_LA_eval",
            "flac"
        )
    }
}

# --------------------------------------------------
# Sampling configuration (balanced dataset)
# --------------------------------------------------

TOTAL_SAMPLES = 65000
TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

PER_CLASS = TOTAL_SAMPLES // 2

TRAIN_PER_CLASS = int(PER_CLASS * TRAIN_RATIO)
DEV_PER_CLASS = int(PER_CLASS * DEV_RATIO)
TEST_PER_CLASS = PER_CLASS - TRAIN_PER_CLASS - DEV_PER_CLASS

QUOTAS = {
    "train": TRAIN_PER_CLASS,
    "dev": DEV_PER_CLASS,
    "test": TEST_PER_CLASS
}

# --------------------------------------------------
# Create directories
# --------------------------------------------------

def create_dirs():

    for split in ["train", "dev", "test"]:
        for cls in ["real", "fake"]:
            path = os.path.join(DEST_RAW, split, cls)
            os.makedirs(path, exist_ok=True)

    os.makedirs(DEST_PROTO, exist_ok=True)

# --------------------------------------------------
# Read protocol
# --------------------------------------------------

def read_protocol(proto_path):

    real = []
    fake = []

    with open(proto_path, "r") as f:
        for line in f:

            parts = line.strip().split()

            filename = parts[1]
            label = parts[-1]

            if label == "bonafide":
                real.append(filename)
            else:
                fake.append(filename)

    return real, fake

# --------------------------------------------------
# Copy files
# --------------------------------------------------

def copy_files(files, src_dir, dst_dir):

    count = 0

    for fname in tqdm(files):

        src = os.path.join(src_dir, fname + ".flac")
        dst = os.path.join(dst_dir, fname + ".flac")

        if os.path.exists(src):
            shutil.copy2(src, dst)
            count += 1
        else:
            print("Missing:", src)

    return count

# --------------------------------------------------
# Process split
# --------------------------------------------------

def process_split(split):

    print(f"\nProcessing {split.upper()}")

    proto = DATA_MAP[split]["proto"]
    audio = DATA_MAP[split]["audio"]

    real_files, fake_files = read_protocol(proto)

    random.shuffle(real_files)
    random.shuffle(fake_files)

    quota = QUOTAS[split]

    real_selected = real_files[:quota]
    fake_selected = fake_files[:quota]

    print("Real:", len(real_selected))
    print("Fake:", len(fake_selected))

    real_dst = os.path.join(DEST_RAW, split, "real")
    fake_dst = os.path.join(DEST_RAW, split, "fake")

    print("Copying real...")
    copy_files(real_selected, audio, real_dst)

    print("Copying fake...")
    copy_files(fake_selected, audio, fake_dst)

    # Save protocol
    proto_out = os.path.join(DEST_PROTO, f"{split}_protocol.txt")

    with open(proto_out, "w") as f:

        for x in real_selected:
            f.write(f"{x} bonafide\n")

        for x in fake_selected:
            f.write(f"{x} spoof\n")

# --------------------------------------------------
# Main
# --------------------------------------------------

def setup_data():

    print("🚀 Preparing dataset")

    create_dirs()

    for split in ["train", "dev", "test"]:
        process_split(split)

    print("\n✅ Dataset ready!")

# --------------------------------------------------

if __name__ == "__main__":
    setup_data()