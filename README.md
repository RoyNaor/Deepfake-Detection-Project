# FusionGuardNet: Audio Deepfake Detection via Semantic & Acoustic Fusion

A multi-modal deep learning system for audio deepfake detection that fuses acoustic and linguistic representations to achieve **99.18% test accuracy** on the ASVspoof 2019 Logical Access benchmark.

---

## Overview

Modern text-to-speech systems are becoming acoustically indistinguishable from real speech, rendering purely signal-based detectors increasingly fragile. FusionGuardNet addresses this by combining two complementary views of an audio signal:

- **Acoustic domain** — signal-level artifacts and frequency anomalies captured by WavLM
- **Semantic/linguistic domain** — prosodic inconsistencies and unnatural phonetic patterns captured by Whisper

The fused representation is fed into a Nes2Net classifier (SOTA 2024) to produce a binary **Bonafide / Spoof** decision.

---

## System Architecture

```
Raw Audio ──┬──► WavLM Backbone  ──► Acoustic Features  ──┐
            │                                              ├──► Learnable Fusion ──► Nes2Net Classifier ──► Real / Fake
            └──► Whisper Backbone ──► Semantic Features  ──┘
```

### Components

| Component | Role | Details |
|-----------|------|---------|
| **WavLM** (Microsoft) | Acoustic encoder | SSL model trained on 94k hours of speech; captures fine-grained signal artifacts |
| **Whisper** (OpenAI) | Semantic encoder | ASR model repurposed to capture prosodic and phonetic inconsistencies |
| **Learnable Weighted Sum Fusion** | Feature fusion | Channel-wise, softmax-gated learnable combination of the two 768-dim feature streams |
| **Nes2Net** | Classifier backend | Lightweight Nested Res2Net architecture; performs the final binary classification |

---

## Results

Trained for 8 epochs on ASVspoof 2019 LA with a balanced 84,278-sample training set.

| Split | Samples | Accuracy | Loss |
|-------|---------|----------|------|
| Train | 84,278 | 99.49% | 0.0186 |
| Dev | 10,534 | **99.25%** | 0.0298 |
| Test | 10,536 | **99.18%** | 0.0324 |

**Test classification report (fake class):**

| Metric | Score |
|--------|-------|
| Precision | 0.9918 |
| Recall | 0.9918 |
| F1 Score | 0.9918 |
| Total Errors | 86 / 10,536 |

**Training configuration:** batch size 16, lr 1e-4, weight decay 1e-4, gradient clipping 1.0, ReduceLROnPlateau scheduler (factor 0.5, patience 2), early stopping patience 4.

---

## Project Structure

```
Deepfake-Detection-Project/
├── models/
│   ├── fusion_guard_net.py      # FusionGuardNet, WavLMBranchNet, WhisperBranchNet
│   └── nes2net_backbone.py      # Nes2Net backend classifier
├── scripts/
│   ├── check_setup.py           # Verify environment and dependencies
│   ├── convert_mpeg.py          # Convert audio files to WAV format
│   ├── download_models.py       # Download WavLM and Whisper checkpoints
│   ├── extract_features.py      # Pre-extract and save WavLM + Whisper features
│   ├── organize_data.py         # Prepare ASVspoof dataset splits
│   ├── train_model.py           # Train FusionGuardNet
│   ├── test_fusion_guard_net.py # Evaluate on the test set
│   └── eval_single_audio.py     # Run inference on a single audio file
├── results/
│   ├── accuracy_curve.png
│   ├── loss_curve.png
│   ├── fusion_guard_net_summary.txt
│   └── test_runs/               # Per-epoch test reports and confusion matrices
├── requirements.txt
└── README.md
```

---

## Installation

**Requirements:** Python 3.8+, CUDA-capable GPU recommended.

```bash
git clone https://github.com/roynaor/deepfake-detection-project.git
cd deepfake-detection-project
pip install -r requirements.txt
```

Verify your setup:

```bash
python scripts/check_setup.py
```

---

## Usage

### 1. Prepare the dataset

Download [ASVspoof 2019 LA](https://datashare.ed.ac.uk/handle/10283/3336), then:

```bash
python scripts/organize_data.py --data_root /path/to/asvspoof2019
```

### 2. Download pre-trained backbones

```bash
python scripts/download_models.py
```

### 3. Pre-extract features

```bash
python scripts/extract_features.py --data_root /path/to/asvspoof2019 --out_dir /path/to/processed
```

### 4. Train

```bash
python scripts/train_model.py --processed_dir /path/to/processed --epochs 8
```

### 5. Evaluate on the test set

```bash
python scripts/test_fusion_guard_net.py --processed_dir /path/to/processed --epoch 8
```

### 6. Run inference on a single file

```bash
python scripts/eval_single_audio.py --audio path/to/audio.wav --checkpoint path/to/checkpoint.pth
```

---

## References

- **Nes2Net:** Liu et al., *"Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing"*, 2024.
- **WavLM:** Chen et al., *"WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing"*, Microsoft Research.
- **Whisper:** Radford et al., *"Robust Speech Recognition via Large-Scale Weak Supervision"*, OpenAI.
- **Dataset:** ASVspoof 2019 Logical Access Challenge.
