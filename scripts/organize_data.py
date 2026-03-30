import os
import random
import shutil
from tqdm import tqdm

# ==================================================
# CONFIG
# ==================================================

ROOT = r"D:\Fusion_Model"

PROJECT_ROOT = os.path.join(ROOT, "dataset")

DEST_RAW = os.path.join(PROJECT_ROOT, "data", "raw")
DEST_PROTO = os.path.join(PROJECT_ROOT, "data", "protocols")

USE_HARDLINKS = True

TOTAL_SAMPLES = 130000

TRAIN_RATIO = 0.8
DEV_RATIO = 0.1
TEST_RATIO = 0.1

RANDOM_SEED = 42

# ==================================================
# PATHS - 2019
# ==================================================

LA_ROOT = os.path.join(ROOT, "LA")

LA_PROTOCOLS = {
    "train": os.path.join(
        LA_ROOT,
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.train.trn.txt",
    ),
    "dev": os.path.join(
        LA_ROOT,
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.dev.trl.txt",
    ),
    "eval": os.path.join(
        LA_ROOT,
        "ASVspoof2019_LA_cm_protocols",
        "ASVspoof2019.LA.cm.eval.trl.txt",
    ),
}

LA_AUDIO_DIRS = [
    os.path.join(LA_ROOT, "ASVspoof2019_LA_train", "flac"),
    os.path.join(LA_ROOT, "ASVspoof2019_LA_dev", "flac"),
    os.path.join(LA_ROOT, "ASVspoof2019_LA_eval", "flac"),
]

# ==================================================
# PATHS - 2024 / ASVspoof5
# ==================================================

PROTO_2024_DIR = os.path.join(ROOT, "ASVspoof5_protocols")

PROTO_2024 = {
    "train": os.path.join(PROTO_2024_DIR, "ASVspoof5.train.tsv"),
    "dev": os.path.join(PROTO_2024_DIR, "ASVspoof5.dev.track_1.tsv"),
}

# ==================================================
# HELPERS
# ==================================================

def create_dirs():
    for split in ["train", "dev", "test"]:
        for cls in ["real", "fake"]:
            os.makedirs(os.path.join(DEST_RAW, split, cls), exist_ok=True)
    os.makedirs(DEST_PROTO, exist_ok=True)


def normalize_label(label: str):
    label = label.strip().lower()

    if label in {"bonafide", "bona-fide", "real", "genuine", "human"}:
        return "bonafide"

    if label in {"spoof", "fake", "synthetic"}:
        return "spoof"

    return None


def discover_asvspoof5_audio_dirs(root_dir):
    dirs = []

    if not os.path.exists(root_dir):
        return dirs

    for item in os.listdir(root_dir):
        full = os.path.join(root_dir, item)
        if os.path.isdir(full) and item.startswith("flac_"):
            dirs.append(full)

    return sorted(dirs)


def build_audio_index(audio_dirs):
    audio_index = {}
    total = 0

    print("\n🔎 Building audio index...")
    for base_dir in audio_dirs:
        if not os.path.exists(base_dir):
            print(f"   [skip] missing dir: {base_dir}")
            continue

        print(f"   scanning: {base_dir}")
        for root, _, files in os.walk(base_dir):
            for file in files:
                if file.lower().endswith(".flac"):
                    utt_id = os.path.splitext(file)[0]
                    full_path = os.path.join(root, file)

                    if utt_id not in audio_index:
                        audio_index[utt_id] = full_path
                        total += 1

    print(f"   indexed {total:,} unique .flac files")
    return audio_index


def parse_2019_protocol(proto_path):
    entries = []

    if not os.path.exists(proto_path):
        print(f"⚠ Missing 2019 protocol: {proto_path}")
        return entries

    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 2:
                continue

            utt_id = parts[1]
            label = normalize_label(parts[-1])

            if label is None:
                continue

            entries.append((utt_id, label))

    return entries


def parse_2024_protocol(proto_path):
    """
    לפי הדוגמאות ששלחת:
    TRAIN:
      T_4850 T_0000000000 F - - - AC3 A05 spoof -
    DEV:
      D_0062 D_0000000001 F - - - AC1 A11 spoof -
      D_0461 D_0000000190 M - - - bonafide bonafide -

    לכן:
      utt_id = parts[1]
      label  = parts[-2]
    """
    entries = []

    if not os.path.exists(proto_path):
        print(f"⚠ Missing 2024 protocol: {proto_path}")
        return entries

    with open(proto_path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()

            if len(parts) < 3:
                continue

            utt_id = parts[1]
            label = normalize_label(parts[-2])

            if label is None:
                continue

            entries.append((utt_id, label))

    return entries


def safe_link_or_copy(src, dst):
    if os.path.exists(dst):
        return

    if USE_HARDLINKS:
        try:
            os.link(src, dst)
            return
        except Exception:
            pass

    shutil.copy2(src, dst)


def write_protocol_file(path, entries):
    with open(path, "w", encoding="utf-8") as f:
        for e in entries:
            f.write(
                f"{e['utt_id']}\t{e['class_name']}\t{e['source_dataset']}\t{e['original_split']}\n"
            )


def write_summary(path, selected_entries):
    by_source = {}
    by_class = {"real": 0, "fake": 0}

    for e in selected_entries:
        by_class[e["class_name"]] += 1
        key = (e["source_dataset"], e["class_name"])
        by_source[key] = by_source.get(key, 0) + 1

    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Total files: {len(selected_entries)}\n")
        f.write(f"Real: {by_class['real']}\n")
        f.write(f"Fake: {by_class['fake']}\n\n")

        f.write("By source:\n")
        for key, value in sorted(by_source.items()):
            f.write(f"{key[0]} | {key[1]} = {value}\n")


# ==================================================
# COLLECT 2019
# ==================================================

def collect_2019_entries():
    print("\n====================")
    print("Collecting 2019 LA")
    print("====================")

    audio_index = build_audio_index(LA_AUDIO_DIRS)
    entries = []

    for original_split, proto_path in LA_PROTOCOLS.items():
        proto_entries = parse_2019_protocol(proto_path)
        print(f"   2019 {original_split}: protocol rows = {len(proto_entries):,}")

        missing = 0
        kept = 0

        for utt_id, label in proto_entries:
            src_path = audio_index.get(utt_id)
            if src_path is None:
                missing += 1
                continue

            class_name = "real" if label == "bonafide" else "fake"

            entries.append({
                "utt_id": utt_id,
                "label": label,
                "class_name": class_name,
                "src_path": src_path,
                "source_dataset": "2019_LA",
                "original_split": original_split,
            })
            kept += 1

        print(f"      kept = {kept:,}, missing_audio = {missing:,}")

    return entries


# ==================================================
# COLLECT 2024
# ==================================================

def collect_2024_entries():
    print("\n====================")
    print("Collecting 2024 ASVspoof5")
    print("====================")

    audio_dirs = discover_asvspoof5_audio_dirs(ROOT)

    print("   discovered ASVspoof5 audio dirs:")
    for d in audio_dirs:
        print(f"      {d}")

    audio_index = build_audio_index(audio_dirs)
    entries = []

    for original_split, proto_path in PROTO_2024.items():
        proto_entries = parse_2024_protocol(proto_path)
        print(f"   2024 {original_split}: protocol rows = {len(proto_entries):,}")

        missing = 0
        kept = 0

        for utt_id, label in proto_entries:
            src_path = audio_index.get(utt_id)
            if src_path is None:
                missing += 1
                continue

            class_name = "real" if label == "bonafide" else "fake"

            entries.append({
                "utt_id": utt_id,
                "label": label,
                "class_name": class_name,
                "src_path": src_path,
                "source_dataset": "2024_ASVspoof5",
                "original_split": original_split,
            })
            kept += 1

        print(f"      kept = {kept:,}, missing_audio = {missing:,}")

    return entries


# ==================================================
# BALANCED RESPLIT
# ==================================================

def compute_target_counts(total_samples):
    per_class = total_samples // 2

    train_per_class = int(per_class * TRAIN_RATIO)
    dev_per_class = int(per_class * DEV_RATIO)
    test_per_class = per_class - train_per_class - dev_per_class

    return {
        "train": train_per_class,
        "dev": dev_per_class,
        "test": test_per_class,
    }


def split_balanced(entries):
    real_entries = [e for e in entries if e["class_name"] == "real"]
    fake_entries = [e for e in entries if e["class_name"] == "fake"]

    random.shuffle(real_entries)
    random.shuffle(fake_entries)

    max_balanced_total = 2 * min(len(real_entries), len(fake_entries))

    if TOTAL_SAMPLES is None:
        final_total = max_balanced_total
    else:
        final_total = min(TOTAL_SAMPLES, max_balanced_total)

    if final_total < 2:
        raise RuntimeError("Not enough balanced data to create dataset.")

    targets = compute_target_counts(final_total)
    per_class_total = final_total // 2

    real_entries = real_entries[:per_class_total]
    fake_entries = fake_entries[:per_class_total]

    real_train = real_entries[:targets["train"]]
    real_dev = real_entries[targets["train"]:targets["train"] + targets["dev"]]
    real_test = real_entries[targets["train"] + targets["dev"]:targets["train"] + targets["dev"] + targets["test"]]

    fake_train = fake_entries[:targets["train"]]
    fake_dev = fake_entries[targets["train"]:targets["train"] + targets["dev"]]
    fake_test = fake_entries[targets["train"] + targets["dev"]:targets["train"] + targets["dev"] + targets["test"]]

    split_map = {
        "train": real_train + fake_train,
        "dev": real_dev + fake_dev,
        "test": real_test + fake_test,
    }

    for split in split_map:
        random.shuffle(split_map[split])

    return split_map, final_total


# ==================================================
# MATERIALIZE
# ==================================================

def materialize_split(split_name, entries):
    print(f"\n📦 Materializing {split_name.upper()}")

    real_dst = os.path.join(DEST_RAW, split_name, "real")
    fake_dst = os.path.join(DEST_RAW, split_name, "fake")

    real_count = sum(1 for e in entries if e["class_name"] == "real")
    fake_count = sum(1 for e in entries if e["class_name"] == "fake")

    print(f"   total = {len(entries):,}")
    print(f"   real  = {real_count:,}")
    print(f"   fake  = {fake_count:,}")

    for e in tqdm(entries, desc=f"{split_name}"):
        dst_dir = real_dst if e["class_name"] == "real" else fake_dst
        dst_path = os.path.join(dst_dir, e["utt_id"] + ".flac")
        safe_link_or_copy(e["src_path"], dst_path)

    protocol_path = os.path.join(DEST_PROTO, f"{split_name}_protocol.txt")
    summary_path = os.path.join(DEST_PROTO, f"{split_name}_summary.txt")

    write_protocol_file(protocol_path, entries)
    write_summary(summary_path, entries)


# ==================================================
# MAIN
# ==================================================

def setup_data():
    random.seed(RANDOM_SEED)

    print("🚀 Preparing combined dataset")
    print(f"ROOT: {ROOT}")
    print(f"PROJECT_ROOT: {PROJECT_ROOT}")
    print(f"USE_HARDLINKS: {USE_HARDLINKS}")
    print(f"TOTAL_SAMPLES requested: {TOTAL_SAMPLES}")

    create_dirs()

    entries_2019 = collect_2019_entries()
    entries_2024 = collect_2024_entries()

    all_entries = entries_2019 + entries_2024

    print("\n====================")
    print("Collected totals")
    print("====================")
    print(f"2019 entries: {len(entries_2019):,}")
    print(f"2024 entries: {len(entries_2024):,}")
    print(f"ALL  entries: {len(all_entries):,}")

    total_real = sum(1 for e in all_entries if e["class_name"] == "real")
    total_fake = sum(1 for e in all_entries if e["class_name"] == "fake")

    print(f"real available: {total_real:,}")
    print(f"fake available: {total_fake:,}")

    split_map, final_total = split_balanced(all_entries)

    print("\n====================")
    print("Final split sizes")
    print("====================")
    print(f"Total balanced dataset size: {final_total:,}")

    for split_name in ["train", "dev", "test"]:
        split_entries = split_map[split_name]
        real_count = sum(1 for e in split_entries if e["class_name"] == "real")
        fake_count = sum(1 for e in split_entries if e["class_name"] == "fake")

        print(
            f"{split_name}: total={len(split_entries):,}, "
            f"real={real_count:,}, fake={fake_count:,}"
        )

    for split_name in ["train", "dev", "test"]:
        materialize_split(split_name, split_map[split_name])

    print("\nDone. Dataset created successfully.")
    print(f"Saved under: {PROJECT_ROOT}")


if __name__ == "__main__":
    setup_data()