# Deepfake Detection Project Analysis

## Overview
This project implements a proof-of-concept pipeline for detecting deepfake audio using pretrained speech backbones (WavLM and Whisper) and a lightweight classifier (Nes2Net). The workflow spans dataset organization, feature extraction, model training, and evaluation. It supports three modes: WavLM-only, Whisper-only, and fused WavLM+Whisper features.

## High-Level Workflow Diagram
1. **Prepare dataset** (`scripts/organize_data.py`)
   - Validate ASVspoof protocol/audio paths.
   - Split bonafide/spoof file lists into train/test subsets.
   - Copy audio files into `data/raw_audio/{train,test}` and emit new protocol files.
2. **Download models** (`scripts/download_models.py`)
   - Cache WavLM and Whisper weights via HuggingFace for offline usage.
3. **Extract features** (`scripts/extract_features.py`)
   - Load WavLM + Whisper models and feature extractor.
   - For each audio file, generate WavLM hidden states and Whisper encoder hidden states.
   - Save per-file tensors in `data/feats_wavlm` and `data/feats_whisper`.
4. **Train & evaluate** (`scripts/main_experiment.py`)
   - Load protocol files and per-file features.
   - Optionally fuse WavLM and Whisper features.
   - Train Nes2Net classifier for each mode and report accuracy.
5. **End-to-end fusion model** (`models/fusion_system.py`)
   - Optional runtime fusion using WavLM + Whisper backbones inside a single model.

## Key Components & Responsibilities
- **`models/nes2net.py`**: Implements the Nes2Net classifier, including Res2Net-style bottlenecks, optional attentive pooling (ASTP), and a binary classification head.
- **`models/fusion_system.py`**: Wraps WavLM and/or Whisper backbones and feeds their features into Nes2Net.
- **`scripts/organize_data.py`**: Splits and copies ASVspoof data into project-local folders and emits train/test protocol files.
- **`scripts/extract_features.py`**: Runs offline feature extraction with WavLM and Whisper to build cached tensors for training.
- **`scripts/main_experiment.py`**: Loads features, trains Nes2Net, and evaluates accuracy for three modes.
- **`scripts/check_setup.py`**: Validates environment (GPU + dependencies) before running heavier jobs.

## Data Flow Summary
- Raw audio (`.flac`) is copied into `data/raw_audio/{train,test}`.
- Features are extracted and saved into `data/feats_wavlm` and `data/feats_whisper` as `.pt` tensors.
- The training script loads the protocol files to map filenames → labels, then loads the corresponding feature tensors and performs training.

## Detected Issues / Risks
1. **Whisper encoder input mismatch in `FusionSystem`**
   - The fusion model passes raw audio directly to `WhisperModel.encoder`, but Whisper expects log-mel spectrograms. This is likely to crash or generate invalid outputs when `FusionSystem` is used directly.
2. **Silent error masking in `FeatureDataset.__getitem__`**
   - Any exception during feature loading returns a zero tensor without logging, which can silently degrade training and make debugging difficult.
3. **Potential zero-division in evaluation**
   - If the test set is empty (e.g., missing protocol file or no files copied), `test_total` becomes zero and the accuracy calculation will raise a division error.
4. **Dataset split sizes not validated**
   - If the requested counts exceed available data, slicing will return smaller arrays and train/test sets may overlap or be severely undersized.
5. **Multi-channel audio handling**
   - `extract_features.py` uses `waveform.squeeze()` before Whisper processing; multi-channel inputs could be collapsed in unintended ways, leading to channel mixing or shape issues.

## Improvement Suggestions (Behavior-Preserving)
- Validate dataset size before slicing in `organize_data.py`, and emit warnings when requested counts exceed available files.
- Log exceptions in `FeatureDataset.__getitem__` so missing or corrupted features are visible.
- Guard against empty test loaders before computing accuracy to avoid division errors.
- Align `FusionSystem` Whisper input with the same preprocessing path used in `extract_features.py` (i.e., use the Whisper feature extractor).
- Add light schema checks (expected tensor shapes) after feature loading to catch mismatches early.

## Assumptions & Limitations
- The ASVspoof2019 LA directory structure is assumed exactly as in the official dataset.
- The pipeline assumes audio can be resampled to 16 kHz and that 4 seconds is representative for WavLM extraction.
- Fixed-length truncation/padding to 200 frames may limit model performance and assumes that downstream Nes2Net expects a fixed temporal length.
- Training/evaluation scripts assume features were precomputed and stored in the expected directory structure.
