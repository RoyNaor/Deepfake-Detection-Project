"""Prepare train/test splits and protocol files for the ASVspoof dataset."""

import os
import shutil
import random

# --- Configuration (update SOURCE_ROOT before running) ---

# 1) Location of the extracted ASVspoof2019 LA folder (absolute path).
# NOTE: This should point to the root folder created after extraction.
SOURCE_ROOT = r"C:\Users\amitn\Downloads\LA"

# 2) Location of this project (default: current directory).
PROJECT_ROOT = r"./"

# 3) Requested dataset sizes for each split.
NUM_TRAIN_REAL = 2000
NUM_TRAIN_FAKE = 2000
NUM_TEST_REAL = 500
NUM_TEST_FAKE = 500

# --- Internal paths (do not change unless you know what you're doing) ---
ORIGINAL_PROTO = os.path.join(
    SOURCE_ROOT,
    "ASVspoof2019_LA_cm_protocols",
    "ASVspoof2019.LA.cm.train.trn.txt",
)
ORIGINAL_AUDIO = os.path.join(SOURCE_ROOT, "ASVspoof2019_LA_train", "flac")

# Output locations for copied audio and generated protocol files.
DEST_TRAIN_DIR = os.path.join(PROJECT_ROOT, "data", "raw_audio", "train")
DEST_TEST_DIR = os.path.join(PROJECT_ROOT, "data", "raw_audio", "test")
DEST_PROTO_DIR = os.path.join(PROJECT_ROOT, "data", "protocols")


def setup_data():
    """Create train/test folders and protocol files from the ASVspoof dataset.

    This function reads the official protocol, selects fixed-size train/test
    subsets, copies the referenced audio files into project-local directories,
    and writes new protocol files that map filenames to labels.

    Side Effects:
        - Creates directories under ``data/raw_audio`` and ``data/protocols``.
        - Copies audio files from SOURCE_ROOT into project-local folders.
        - Writes protocol text files for train/test splits.
    """
    print("--- Starting dataset organization ---")

    # 1) Verify that required source files exist before proceeding.
    if not os.path.exists(ORIGINAL_PROTO):
        print(f"Error: Protocol file not found at:\n{ORIGINAL_PROTO}")
        return
    if not os.path.exists(ORIGINAL_AUDIO):
        print(f"Error: Audio directory not found at:\n{ORIGINAL_AUDIO}")
        return

    # 2) Create output directories (idempotent).
    os.makedirs(DEST_TRAIN_DIR, exist_ok=True)
    os.makedirs(DEST_TEST_DIR, exist_ok=True)
    os.makedirs(DEST_PROTO_DIR, exist_ok=True)

    # 3) Read the protocol and separate bonafide vs spoof entries.
    print("Reading source protocol file...")
    real_files = []
    fake_files = []

    with open(ORIGINAL_PROTO, "r") as f:
        for line in f:
            parts = line.strip().split()
            filename = parts[1]
            label = parts[-1]  # 'bonafide' or 'spoof'

            if label == "bonafide":
                real_files.append(filename)
            else:
                fake_files.append(filename)

    # Randomize to reduce ordering bias before slicing into splits.
    random.shuffle(real_files)
    random.shuffle(fake_files)

    print(
        f"Found {len(real_files)} bonafide and {len(fake_files)} spoof files in total."
    )

    # 4) Split into train/test subsets.
    train_real = real_files[:NUM_TRAIN_REAL]
    train_fake = fake_files[:NUM_TRAIN_FAKE]

    # Use the tail of the list for test to minimize overlap with train.
    test_real = real_files[-NUM_TEST_REAL:]
    test_fake = fake_files[-NUM_TEST_FAKE:]

    print(
        f"Selected for training: {len(train_real)} bonafide, {len(train_fake)} spoof."
    )
    print(f"Selected for testing: {len(test_real)} bonafide, {len(test_fake)} spoof.")

    # 5) Helper to copy files and emit a new protocol file for each split.
    def process_subset(file_list, dest_dir, protocol_name, label_map):
        """Copy audio files and write a split-specific protocol file.

        Args:
            file_list: Ordered list of file IDs (without extension).
            dest_dir: Destination directory for copied audio files.
            protocol_name: Output protocol filename.
            label_map: Mapping with a "real" set used to assign labels.

        Side Effects:
            - Copies audio files into dest_dir.
            - Writes a protocol file to DEST_PROTO_DIR.
        """
        new_proto_path = os.path.join(DEST_PROTO_DIR, protocol_name)

        print(f"Copying files into {dest_dir}...")
        with open(new_proto_path, "w") as f_proto:
            for fname in file_list:
                # Copy the audio file if it exists.
                src_path = os.path.join(ORIGINAL_AUDIO, fname + ".flac")
                dst_path = os.path.join(dest_dir, fname + ".flac")

                try:
                    shutil.copy2(src_path, dst_path)

                    # Write label in simple "filename label" format.
                    is_real = 1 if fname in label_map["real"] else 0
                    label_str = "bonafide" if is_real else "spoof"

                    f_proto.write(f"{fname} {label_str}\n")

                except FileNotFoundError:
                    print(f"Warning: File {fname} not found via path {src_path}")

    # Build the training split and protocol.
    train_all = train_real + train_fake
    random.shuffle(train_all)  # Shuffle again to mix labels in the protocol.

    # Use a set for O(1) membership checks when assigning labels.
    real_set = set(real_files)
    label_map = {"real": real_set}

    process_subset(train_all, DEST_TRAIN_DIR, "train_protocol.txt", label_map)

    # Build the testing split and protocol.
    test_all = test_real + test_fake
    random.shuffle(test_all)
    process_subset(test_all, DEST_TEST_DIR, "test_protocol.txt", label_map)

    print("\n--- Dataset organization complete ---")
    print("Copied audio files to data/raw_audio.")
    print("Generated new protocol files in data/protocols.")


if __name__ == "__main__":
    setup_data()
