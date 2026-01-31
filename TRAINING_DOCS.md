
## 1. Experiment Overview
This project evaluates the effectiveness of fusing **Semantic Features** (Whisper) with **Acoustic Features** (WavLM) for detecting deepfake audio. The core hypothesis is that combining these two modalities improves robustness against various spoofing attacks compared to using a single modality.

**Architecture:**
- **Backbone A (Acoustic):** WavLM Base Plus (Frozen)
- **Backbone B (Semantic):** Whisper Small (Frozen)
- **Classifier (Head):** Nes2Net (Trainable)

---

## 2. Dataset Configuration (POC Scope)
To ensure rapid iteration and balanced training for the Proof-of-Concept (POC), we utilized a subset of the **ASVspoof 2019 Logical Access (LA)** dataset.

### Data Distribution
We curated a balanced dataset to prevent model bias:

| Split | Real (Bonafide) | Fake (Spoof) | Total Files | Description |
| :--- | :---: | :---: | :---: | :--- |
| **Train** | 2,000 | 2,000 | **4,000** | Used for training the Nes2Net classifier. |
| **Test** | 500 | 500 | **1,000** | Unseen data used for final evaluation. |
| **Total** | **2,500** | **2,500** | **5,000** | Total files processed. |

* **Source:** ASVspoof 2019 LA Train Partition.
* **Selection Method:** Random balanced sampling.

---

## 3. Pre-processing & Feature Extraction

Due to the high computational cost of running two foundation models simultaneously, features were **pre-extracted** and stored offline.

### Audio Settings
* **Sampling Rate:** 16kHz (Resampled from original if necessary).
* **Channels:** Mono (Stereo files converted to mono).
* **WavLM Input:** Raw waveform, truncated/padded to **4 seconds** (64,000 samples).
* **Whisper Input:** Log-Mel Spectrogram (via `AutoFeatureExtractor`), padded to **30 seconds**.

### Feature Dimensions
The extracted feature tensors (`.pt` files) have the following shapes:

| Model | Output Dimension | Notes |
| :--- | :--- | :--- |
| **WavLM** | `[Time, 768]` | Extracts acoustic/speaker artifacts. |
| **Whisper** | `[Time, 768]` | Extracts phonetic/linguistic anomalies. |
| **Fusion** | `[Time, 1536]` | Concatenation of aligned WavLM + Whisper features. |

*Note: For the fusion experiment, Whisper features are interpolated (resized) to match the temporal length of WavLM features before concatenation.*

---

## 4. Training Hyperparameters

The Nes2Net classifier was trained using the following configuration:

* **Environment:** PyTorch (CUDA/GPU)
* **Batch Size:** 16
* **Epochs:** 10 (Sufficient for convergence on POC subset)
* **Learning Rate:** 0.0001 (`1e-4`)
* **Optimizer:** Adam
* **Loss Function:** CrossEntropyLoss
* **Input Length:** Fixed to 200 frames (Truncation/Padding applied during data loading).

---

## 5. Directory Structure
The system expects the following directory layout:

```text
Deepfake_Project/
├── data/
│   ├── raw_audio/          # 5,000 FLAC files (Train + Test)
│   ├── protocols/          # Text files mapping filenames to labels
│   ├── feats_wavlm/        # Pre-computed .pt tensors
│   └── feats_whisper/      # Pre-computed .pt tensors
├── checkpoints/            # Saved "best_model.pth"
├── results/                # Accuracy logs and plots
└── scripts/                # Python execution scripts
