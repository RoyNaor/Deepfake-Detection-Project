import torch
import torchaudio
import os
import numpy as np
from tqdm import tqdm
from transformers import WavLMModel, WhisperModel, WhisperProcessor

# -----------------------------
# Config
# -----------------------------

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAMPLE_RATE = 16000
DURATION = 4  # Audio length inseconds
MAX_SAMPLES = SAMPLE_RATE * DURATION

RAW_AUDIO_DIR = "./data/raw"
PROCESSED_AUDIO_DIR = "./data/processed"


# -----------------------------
# Feature Extraction
# -----------------------------

def get_aligned_features(waveform, model_wavlm, model_whisper, processor):
    """
    Extracts:
    - WavLM features
    - Whisper encoder features
    - WavLM mask
    - Whisper mask
    """

    num_samples = waveform.shape[1]

    # -----------------------------
    # 1️⃣ Crop / Pad to 4 seconds
    # -----------------------------
    if num_samples > MAX_SAMPLES:
        diff = num_samples - MAX_SAMPLES
        start = torch.randint(0, diff, (1,)).item()
        waveform = waveform[:, start : start + MAX_SAMPLES]
        real_ratio = 1.0
    else:
        pad_len = MAX_SAMPLES - num_samples
        waveform = torch.nn.functional.pad(waveform, (0, pad_len))
        real_ratio = num_samples / MAX_SAMPLES

    wav_input = waveform.to(DEVICE)

    # -----------------------------
    # 2️⃣ WavLM Extraction
    # -----------------------------
    with torch.no_grad():
        wavlm_out = model_wavlm(wav_input)
        wavlm_feats = wavlm_out.last_hidden_state  # [B, T_wav, 768]

    T_wav = wavlm_feats.shape[1]

    # Build WavLM mask
    real_frames_wav = int(real_ratio * T_wav)
    wavlm_mask = torch.cat([
        torch.ones(real_frames_wav),
        torch.zeros(T_wav - real_frames_wav)
    ])

    # -----------------------------
    # 3️⃣ Whisper Extraction
    # -----------------------------
    whisper_inputs = processor(
        waveform.squeeze().cpu().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    whisper_input_features = whisper_inputs.input_features.to(DEVICE)

    with torch.no_grad():
        whisper_out = model_whisper.encoder(whisper_input_features)
        whisper_feats = whisper_out.last_hidden_state  # [B, T_whisper, 768]

    T_whisper = whisper_feats.shape[1]

    # Build Whisper mask
    real_frames_whisper = int(real_ratio * T_whisper)
    whisper_mask = torch.cat([
        torch.ones(real_frames_whisper),
        torch.zeros(T_whisper - real_frames_whisper)
    ])

    # FIX: Whisper auto-pads audio to 30s (1500 frames). 
    # Since both WavLM and Whisper output at exactly 50Hz, interpolation squashes 30s into 4s and ruins the alignment.
    # Instead, we simply slice Whisper's output to match WavLM's exact length (T_wav).
    # when using this- delete step 4 and put this code instead the original end of step 3.
    
    # with torch.no_grad():
    #     whisper_out = model_whisper.encoder(whisper_input_features)
    #     whisper_feats = whisper_out.last_hidden_state  # [B, 1500, 768]

    # # --- 4️⃣ Align Time Dimensions & Masks (FIXED) ---
    
    # # Slice Whisper to match exactly the 4 seconds (T_wav)
    # # This throws away the 26 seconds of padded silence Whisper automatically added
    # whisper_feats = whisper_feats[:, :T_wav, :] 
    
    # # Since both operate at the exact same time resolution (50Hz),
    # # the mask for Whisper is exactly the same as the mask for WavLM!
    # whisper_mask = wavlm_mask.clone()

    # return (
    #     wavlm_feats.cpu(),
    #     whisper_feats.cpu(),
    #     wavlm_mask,
    #     whisper_mask
    # )

    # -----------------------------
    # 4️⃣ Align Time Dimensions
    # -----------------------------
    if T_wav != T_whisper:
        whisper_feats = torch.nn.functional.interpolate(
            whisper_feats.transpose(1, 2),
            size=T_wav,
            mode="linear",
            align_corners=False
        ).transpose(1, 2)

        # Also interpolate mask
        whisper_mask = torch.nn.functional.interpolate(
            whisper_mask.unsqueeze(0).unsqueeze(0),
            size=T_wav,
            mode="nearest"
        ).squeeze()

    return (
        wavlm_feats.cpu(),
        whisper_feats.cpu(),
        wavlm_mask,
        whisper_mask
    )


# -----------------------------
# Dataset Processing
# -----------------------------

def process_dataset(input_dir, output_dir, label,
                    model_wavlm, model_whisper, processor):

    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir)
             if f.endswith(('.flac', '.wav'))]

    # Loop through all the audio files
    for fname in tqdm(files, desc=f"Processing {os.path.basename(input_dir)}"):
        path = os.path.join(input_dir, fname)
        waveform, sr = torchaudio.load(path)

        if sr != SAMPLE_RATE:
            waveform = torchaudio.functional.resample(
                waveform, sr, SAMPLE_RATE)

        # Convert to mono
        if waveform.shape[0] > 1:
            waveform = waveform[0:1, :]

        wav_feats, wh_feats, wav_mask, wh_mask = get_aligned_features(
            waveform,
            model_wavlm,
            model_whisper,
            processor
        )

        data = {
            "wavlm": wav_feats.squeeze(0),
            "whisper": wh_feats.squeeze(0),
            "wavlm_mask": wav_mask,
            "whisper_mask": wh_mask,
            "label": label
        }

        out_fname = fname.replace('.flac', '.pt').replace('.wav', '.pt')
        torch.save(data, os.path.join(output_dir, out_fname))


# -----------------------------
# Main
# -----------------------------

def main():
    print(f"🚀 Starting feature extraction on {DEVICE}")

    print("Loading WavLM...")
    wavlm = WavLMModel.from_pretrained(
        "microsoft/wavlm-base-plus"
    ).to(DEVICE).eval()

    print("Loading Whisper...")
    whisper = WhisperModel.from_pretrained(
        "openai/whisper-small"
    ).to(DEVICE).eval()

    # the processor makes the audio into spectograms for the whisper model 
    processor = WhisperProcessor.from_pretrained(
        "openai/whisper-small"
    )

    splits = ["train", "test"]
    classes = {"real": 0, "fake": 1}

    for split in splits:
        for cls_name, cls_label in classes.items():
            input_path = os.path.join(
                RAW_AUDIO_DIR, split, cls_name)
            output_path = os.path.join(
                PROCESSED_AUDIO_DIR, split, cls_name)

            if os.path.exists(input_path):
                process_dataset(
                    input_path,
                    output_path,
                    cls_label,
                    wavlm,
                    whisper,
                    processor
                )
            else:
                print(f"⚠️ Folder not found: {input_path}")

    print("\n✅ Feature extraction completed.")


if __name__ == "__main__":
    main()
