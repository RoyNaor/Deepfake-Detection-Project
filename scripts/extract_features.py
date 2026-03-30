import os
import random
import traceback

import torch
import torchaudio
import torch.nn.functional as F
from tqdm import tqdm
from transformers import WavLMModel, WhisperModel, WhisperProcessor

# ==================================================
# CONFIG
# ==================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
DURATION = 4
MAX_SAMPLES = SAMPLE_RATE * DURATION

DATASET_ROOT = r"D:\Fusion_Model\dataset"
RAW_AUDIO_DIR = os.path.join(DATASET_ROOT, "data", "raw")
PROCESSED_AUDIO_DIR = os.path.join(DATASET_ROOT, "data", "processed")

SPLITS = ["train", "dev", "test"]
CLASSES = {"real": 0, "fake": 1}

RANDOM_SEED = 42

SAVE_EXTRA_METADATA = True

# ==================================================
# HELPERS
# ==================================================

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_audio(path: str):
    waveform, sr = torchaudio.load(path)

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(
            waveform,
            orig_freq=sr,
            new_freq=SAMPLE_RATE
        )
        sr = SAMPLE_RATE

    return waveform, sr


def crop_or_pad_waveform(waveform: torch.Tensor):
    num_samples = waveform.shape[1]

    if num_samples > MAX_SAMPLES:
        diff = num_samples - MAX_SAMPLES
        start = random.randint(0, diff)
        waveform = waveform[:, start:start + MAX_SAMPLES]
        real_ratio = 1.0
    else:
        pad_len = MAX_SAMPLES - num_samples
        waveform = F.pad(waveform, (0, pad_len))
        real_ratio = num_samples / MAX_SAMPLES if MAX_SAMPLES > 0 else 1.0

    return waveform, real_ratio


def build_mask(num_frames: int, real_ratio: float):
    real_frames = int(real_ratio * num_frames)
    real_frames = max(0, min(real_frames, num_frames))

    mask = torch.cat([
        torch.ones(real_frames),
        torch.zeros(num_frames - real_frames)
    ]).float()

    return mask


# ==================================================
# FEATURE EXTRACTION
# ==================================================

def get_aligned_features(waveform, model_wavlm, model_whisper, processor):
    """
    Input:
        waveform: [1, num_samples]

    Output:
        wavlm_feats:    [T, D1]
        whisper_feats:  [T, D2]
        wavlm_mask:     [T]
        whisper_mask:   [T]
    """
    waveform, real_ratio = crop_or_pad_waveform(waveform)

    wav_input = waveform.squeeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        wavlm_out = model_wavlm(wav_input)
        wavlm_feats = wavlm_out.last_hidden_state  # [1, T_wav, C]

    T_wav = wavlm_feats.shape[1]
    wavlm_mask = build_mask(T_wav, real_ratio)

    whisper_inputs = processor(
        waveform.squeeze(0).cpu().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    whisper_input_features = whisper_inputs.input_features.to(DEVICE)

    with torch.no_grad():
        whisper_out = model_whisper.encoder(whisper_input_features)
        whisper_feats = whisper_out.last_hidden_state  # [1, T_wh, C]

    T_wh = whisper_feats.shape[1]

    if T_wh >= T_wav:
        whisper_feats = whisper_feats[:, :T_wav, :]
    else:
        pad_frames = T_wav - T_wh
        whisper_feats = F.pad(whisper_feats, (0, 0, 0, pad_frames))

    whisper_mask = wavlm_mask.clone()

    return (
        wavlm_feats.squeeze(0).cpu(),
        whisper_feats.squeeze(0).cpu(),
        wavlm_mask.cpu(),
        whisper_mask.cpu(),
    )


# ==================================================
# PROCESSING
# ==================================================

def process_single_file(
    input_path: str,
    output_path: str,
    label: int,
    split: str,
    cls_name: str,
    model_wavlm,
    model_whisper,
    processor
):
    waveform, sr = load_audio(input_path)

    wav_feats, wh_feats, wav_mask, wh_mask = get_aligned_features(
        waveform,
        model_wavlm,
        model_whisper,
        processor
    )

    data = {
        "wavlm": wav_feats,
        "whisper": wh_feats,
        "wavlm_mask": wav_mask,
        "whisper_mask": wh_mask,
        "label": label,
    }

    if SAVE_EXTRA_METADATA:
        data["meta"] = {
            "source_path": input_path,
            "split": split,
            "class_name": cls_name,
            "sample_rate": SAMPLE_RATE,
            "duration_sec": DURATION,
            "filename": os.path.basename(input_path),
        }

    torch.save(data, output_path)


def process_dataset_folder(
    input_dir: str,
    output_dir: str,
    label: int,
    split: str,
    cls_name: str,
    model_wavlm,
    model_whisper,
    processor
):
    ensure_dir(output_dir)

    files = sorted([
        f for f in os.listdir(input_dir)
        if f.lower().endswith((".flac", ".wav"))
    ])

    if not files:
        print(f"No audio files found in: {input_dir}")
        return 0, 0

    processed_count = 0
    skipped_count = 0

    for fname in tqdm(files, desc=f"{split}/{cls_name}"):
        out_fname = os.path.splitext(fname)[0] + ".pt"
        out_path = os.path.join(output_dir, out_fname)

        if os.path.exists(out_path):
            skipped_count += 1
            continue

        input_path = os.path.join(input_dir, fname)

        try:
            process_single_file(
                input_path=input_path,
                output_path=out_path,
                label=label,
                split=split,
                cls_name=cls_name,
                model_wavlm=model_wavlm,
                model_whisper=model_whisper,
                processor=processor,
            )
            processed_count += 1

        except Exception as e:
            print(f"\nFailed on file: {input_path}")
            print(f"   Error: {e}")
            traceback.print_exc()

    return processed_count, skipped_count


# ==================================================
# MAIN
# ==================================================

def main():
    set_seed(RANDOM_SEED)

    print(f"🚀 Starting feature extraction on {DEVICE}")
    print(f"RAW_AUDIO_DIR: {RAW_AUDIO_DIR}")
    print(f"PROCESSED_AUDIO_DIR: {PROCESSED_AUDIO_DIR}")

    if not os.path.exists(RAW_AUDIO_DIR):
        raise FileNotFoundError(f"Raw dataset folder not found: {RAW_AUDIO_DIR}")

    ensure_dir(PROCESSED_AUDIO_DIR)

    print("Loading WavLM...")
    wavlm = WavLMModel.from_pretrained(
        "microsoft/wavlm-base-plus"
    ).to(DEVICE).eval()

    print("Loading Whisper...")
    whisper = WhisperModel.from_pretrained(
        "openai/whisper-small"
    ).to(DEVICE).eval()

    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small"
    )

    total_processed = 0
    total_skipped = 0

    for split in SPLITS:
        for cls_name, cls_label in CLASSES.items():
            input_path = os.path.join(RAW_AUDIO_DIR, split, cls_name)
            output_path = os.path.join(PROCESSED_AUDIO_DIR, split, cls_name)

            if not os.path.exists(input_path):
                print(f"Folder not found: {input_path}")
                continue

            print(f"\nProcessing: {input_path}")
            processed_count, skipped_count = process_dataset_folder(
                input_dir=input_path,
                output_dir=output_path,
                label=cls_label,
                split=split,
                cls_name=cls_name,
                model_wavlm=wavlm,
                model_whisper=whisper,
                processor=processor,
            )

            print(
                f"Done {split}/{cls_name} | "
                f"new: {processed_count:,} | skipped existing: {skipped_count:,}"
            )

            total_processed += processed_count
            total_skipped += skipped_count

    print("\n===================================")
    print("Feature extraction completed.")
    print(f"New files processed: {total_processed:,}")
    print(f"Skipped existing:    {total_skipped:,}")
    print(f"Saved under: {PROCESSED_AUDIO_DIR}")
    print("===================================")


if __name__ == "__main__":
    main()