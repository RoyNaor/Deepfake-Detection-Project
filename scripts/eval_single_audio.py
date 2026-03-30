import os
import sys
import math
import torch
import torchaudio
import torch.nn.functional as F

from transformers import WavLMModel, WhisperModel, WhisperProcessor

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

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SAMPLE_RATE = 16000
DURATION = 4
MAX_SAMPLES = SAMPLE_RATE * DURATION
FIXED_LEN = 200

CHECKPOINT_PATH = r"D:\Fusion_Model\dataset\checkpoints\best_fusion_guard_net.pth"

AUDIO_PATH = os.path.join(parent_dir, "demo_files", "testAudio.wav")

# מצב עבודה:
# "single_start"  -> לוקח רק את 4 השניות הראשונות
# "single_random" -> לוקח חלון אקראי אחד של 4 שניות
# "sliding"       -> עובר על כל האודיו עם חלונות
MODE = "sliding"

# רק עבור sliding:
WINDOW_SECONDS = 4
HOP_SECONDS = 2

LABEL_MAP = {0: "real", 1: "fake"}


# --------------------------------------------------
# Utils
# --------------------------------------------------

def load_audio(audio_path):
    waveform, sr = torchaudio.load(audio_path)

    if sr != SAMPLE_RATE:
        waveform = torchaudio.functional.resample(waveform, sr, SAMPLE_RATE)

    if waveform.shape[0] > 1:
        waveform = waveform[:1]

    return waveform


def fix_feature_length(feat, fixed_len=FIXED_LEN):
    if feat.shape[1] > fixed_len:
        feat = feat[:, :fixed_len]
    elif feat.shape[1] < fixed_len:
        pad_amt = fixed_len - feat.shape[1]
        feat = F.pad(feat, (0, pad_amt))
    return feat


def build_windows(waveform, mode="sliding", window_seconds=4, hop_seconds=2):
    num_samples = waveform.shape[1]
    window_size = SAMPLE_RATE * window_seconds
    hop_size = SAMPLE_RATE * hop_seconds

    windows = []

    if mode == "single_start":
        chunk = waveform[:, :window_size]
        if chunk.shape[1] < window_size:
            chunk = F.pad(chunk, (0, window_size - chunk.shape[1]))
        windows.append((0.0, min(num_samples / SAMPLE_RATE, window_seconds), chunk))

    elif mode == "single_random":
        if num_samples <= window_size:
            chunk = waveform
            if chunk.shape[1] < window_size:
                chunk = F.pad(chunk, (0, window_size - chunk.shape[1]))
            windows.append((0.0, num_samples / SAMPLE_RATE, chunk))
        else:
            diff = num_samples - window_size
            start = torch.randint(0, diff + 1, (1,)).item()
            end = start + window_size
            chunk = waveform[:, start:end]
            windows.append((start / SAMPLE_RATE, end / SAMPLE_RATE, chunk))

    elif mode == "sliding":
        if num_samples <= window_size:
            chunk = waveform
            if chunk.shape[1] < window_size:
                chunk = F.pad(chunk, (0, window_size - chunk.shape[1]))
            windows.append((0.0, num_samples / SAMPLE_RATE, chunk))
        else:
            start = 0
            while start + window_size <= num_samples:
                end = start + window_size
                chunk = waveform[:, start:end]
                windows.append((start / SAMPLE_RATE, end / SAMPLE_RATE, chunk))
                start += hop_size

            if len(windows) == 0 or windows[-1][1] * SAMPLE_RATE < num_samples:
                start = max(0, num_samples - window_size)
                end = num_samples
                chunk = waveform[:, start:start + window_size]
                if chunk.shape[1] < window_size:
                    chunk = F.pad(chunk, (0, window_size - chunk.shape[1]))
                windows.append((start / SAMPLE_RATE, end / SAMPLE_RATE, chunk))
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return windows


# --------------------------------------------------
# Feature Extraction
# --------------------------------------------------

def get_aligned_features(window_waveform, model_wavlm, model_whisper, processor):
    """
    Input:
        window_waveform: [1, samples]  (ideally already 4 seconds)
    Output:
        wavlm_feats:   [768, T]
        whisper_feats: [768, T]
    """

    num_samples = window_waveform.shape[1]

    if num_samples > MAX_SAMPLES:
        window_waveform = window_waveform[:, :MAX_SAMPLES]
        real_ratio = 1.0
    else:
        pad_len = MAX_SAMPLES - num_samples
        window_waveform = F.pad(window_waveform, (0, pad_len))
        real_ratio = num_samples / MAX_SAMPLES

    wav_input = window_waveform.squeeze().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        wavlm_out = model_wavlm(wav_input)
        wavlm_feats = wavlm_out.last_hidden_state  # [1, T_wav, 768]

    T_wav = wavlm_feats.shape[1]
    real_frames_wav = int(real_ratio * T_wav)

    # Whisper
    whisper_inputs = processor(
        window_waveform.squeeze().cpu().numpy(),
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt"
    )

    whisper_input_features = whisper_inputs.input_features.to(DEVICE)

    with torch.no_grad():
        whisper_out = model_whisper.encoder(whisper_input_features)
        whisper_feats = whisper_out.last_hidden_state  # [1, T_wh, 768]

    whisper_feats = whisper_feats[:, :T_wav, :]

    wavlm_feats = wavlm_feats.squeeze(0).transpose(0, 1).float().cpu()
    whisper_feats = whisper_feats.squeeze(0).transpose(0, 1).float().cpu()

    wavlm_feats = fix_feature_length(wavlm_feats, FIXED_LEN)
    whisper_feats = fix_feature_length(whisper_feats, FIXED_LEN)

    return wavlm_feats, whisper_feats


# --------------------------------------------------
# Model
# --------------------------------------------------

def load_feature_extractors():
    print("Loading WavLM...")
    wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus").to(DEVICE).eval()

    print("Loading Whisper...")
    whisper = WhisperModel.from_pretrained("openai/whisper-small").to(DEVICE).eval()

    print("Loading Whisper processor...")
    processor = WhisperProcessor.from_pretrained("openai/whisper-small")

    return wavlm, whisper, processor


def load_fusion_model():
    model = FusionGuardNet(
        feature_channels=768,
        nes_ratio=(8, 8),
        dilation=2,
        pool_func="mean",
        se_ratio=(8,),
        num_classes=2,
        use_softmax_gate=True
    ).to(DEVICE)

    checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"Loaded checkpoint from: {CHECKPOINT_PATH}")
    return model


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict_window(fusion_model, wavlm_feat, whisper_feat):
    wavlm_feat = wavlm_feat.unsqueeze(0).to(DEVICE)      # [1, 768, T]
    whisper_feat = whisper_feat.unsqueeze(0).to(DEVICE)  # [1, 768, T]

    with torch.no_grad():
        logits = fusion_model(wavlm_feat, whisper_feat)
        probs = torch.softmax(logits, dim=1).squeeze(0).cpu()

    pred = int(torch.argmax(probs).item())
    return pred, probs


def evaluate_audio_file(audio_path):
    print("=" * 60)
    print("FusionGuardNet Single Audio Evaluation")
    print("=" * 60)
    print(f"Device: {DEVICE}")
    print(f"Audio path: {audio_path}")
    print(f"Mode: {MODE}")

    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    waveform = load_audio(audio_path)
    duration_sec = waveform.shape[1] / SAMPLE_RATE
    print(f"Audio duration: {duration_sec:.2f} seconds")

    wavlm_model, whisper_model, processor = load_feature_extractors()
    fusion_model = load_fusion_model()

    windows = build_windows(
        waveform,
        mode=MODE,
        window_seconds=WINDOW_SECONDS,
        hop_seconds=HOP_SECONDS
    )

    print(f"Number of windows: {len(windows)}")
    print("-" * 60)

    all_probs = []
    all_preds = []

    for i, (start_sec, end_sec, chunk) in enumerate(windows, start=1):
        wavlm_feat, whisper_feat = get_aligned_features(
            chunk,
            wavlm_model,
            whisper_model,
            processor
        )

        pred, probs = predict_window(fusion_model, wavlm_feat, whisper_feat)

        all_probs.append(probs)
        all_preds.append(pred)

        print(
            f"Window {i:02d} | "
            f"{start_sec:.2f}s - {end_sec:.2f}s | "
            f"Pred: {LABEL_MAP[pred]} | "
            f"Real: {probs[0].item():.6f} | "
            f"Fake: {probs[1].item():.6f}"
        )

    mean_probs = torch.stack(all_probs, dim=0).mean(dim=0)
    final_pred = int(torch.argmax(mean_probs).item())

    print("-" * 60)
    print("FINAL RESULT")
    print(f"Prediction: {LABEL_MAP[final_pred]}")
    print(f"Mean prob real: {mean_probs[0].item():.6f}")
    print(f"Mean prob fake: {mean_probs[1].item():.6f}")
    print("=" * 60)


if __name__ == "__main__":
    evaluate_audio_file(AUDIO_PATH)