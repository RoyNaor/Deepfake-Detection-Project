**1\. Introduction**

1.1. Challenges in Audio-Based Deepfake Detection

Recent advances in generative AI have made it significantly easier to produce synthetic speech that sounds indistinguishable from a real human voice. These audio deepfakes pose a growing security threat - they can be used to bypass voice authentication systems, impersonate individuals in phone calls, or spread disinformation through fabricated recordings. As this technology becomes more accessible, the need for reliable automatic detection has become increasingly important.

Building an effective detector is not straightforward. Classical approaches relied on handcrafted audio features that were designed for tasks like speech recognition, and they tend to miss the subtle artifacts that modern synthesis methods introduce. A deeper issue is generalization: a model trained to detect one type of fake audio may fail entirely when faced with a different synthesis method it has not seen before. In practice, new generation tools appear frequently, so a detector that only works on known attack types offers limited real-world protection. Our project addresses this challenge by exploring how combining different types of pre-trained audio representations can produce a more robust and accurate deepfake detector.

---

1.2. Project Goals: Improving Spoofing Classification Using Pre-trained Models

The main goal of this project is to build a system that can reliably distinguish between real and synthetic speech. Rather than designing handcrafted features or training a model from scratch, we chose to leverage large pre-trained models - models that have already learned rich representations of natural speech from massive amounts of audio data. Our reasoning is that these representations capture patterns in speech that are far more informative than what simple acoustic features can express, and that a classifier built on top of them will be better equipped to detect the subtle inconsistencies that fake audio introduces.

Beyond using a single pre-trained model, a key goal was to explore whether combining two models with different strengths could improve detection further. Different types of synthesis errors leave different traces in the audio - some are more signal-level, others relate to how speech sounds at a phonetic or rhythmic level. By giving the classifier a richer, multi-perspective view of the input, we aimed to improve both accuracy and robustness. The end result is a complete, working detection pipeline that we evaluate on a standard benchmark and compare against baseline approaches.

1.3. Our Contribution: Integrating Whisper with WavLM in the Nes2Net Architecture

Our main contribution is the design and implementation of FusionGuardNet, a detection system that combines two pre-trained models - WavLM and Whisper - to extract complementary representations from the same audio input. WavLM is a self-supervised acoustic model trained on large amounts of raw speech; it captures low-level signal patterns and spectral characteristics that reflect how the voice sounds. Whisper, originally built for automatic speech recognition, brings a different perspective by encoding phonetic and prosodic information - how speech is structured at a linguistic level. Both representations are passed through a learnable fusion layer and into a shared Nes2Net classifier, which produces the final real/fake decision.

The motivation behind this design is that synthetic speech can fail in more than one way - some artifacts are acoustic, others are phonetic - and combining two models that look at the signal differently gives the system a better chance of catching both. We evaluate this on the ASVspoof 2019 benchmark and show that the dual-encoder approach achieves 99.18% test accuracy, with equal false positive and false negative rates, outperforming single-model baselines.

**2\. Related Work** 

2.1 Classical and Modern Approaches to Audio Spoofing Detection

Early anti-spoofing systems relied on handcrafted audio features combined with classical machine learning classifiers. These methods captured broad spectral properties of the audio and worked reasonably well against the synthesis systems of their time, which left relatively obvious acoustic traces. As neural speech synthesis improved and produced more natural-sounding output, these approaches became insufficient - their fixed representations were simply not sensitive enough to the subtle differences between real and synthetic speech.

The field gradually moved toward deep learning, which allowed models to learn more discriminative patterns directly from data rather than relying on manually designed features. This improved detection across a range of attack types. The most recent and effective direction has been the use of large self-supervised models pretrained on vast amounts of natural speech. These models produce rich audio representations that capture general properties of human speech, and have been shown to generalize significantly better to new and unseen synthesis methods - making them the foundation of current state-of-the-art detection systems, and the starting point for our own work.

2.2 Pre-trained Speech and ASR Models: Capabilities and Applications of WavLM and Whisper

WavLM is a self-supervised speech model developed by Microsoft, trained on a large and diverse corpus of audio data. During training, it learns to reconstruct masked portions of the audio signal from noisy and clean speech, which forces it to develop a deep understanding of how natural speech is structured at the acoustic level. It was originally designed to perform well across a wide range of speech tasks — including speaker recognition and speech separation — and achieves strong results on most of them without task-specific fine-tuning. For deepfake detection, its key value is that it captures fine-grained acoustic properties that tend to deviate from natural patterns when audio is synthetically generated.

Whisper, developed by OpenAI, takes a fundamentally different approach. It is an automatic speech recognition model trained on a massive amount of transcribed audio collected from diverse sources across the internet. Because its objective is to accurately transcribe spoken content, it learns to encode phonetic and linguistic structure — how speech sounds are organized into words, syllables, and sentences. While it was not designed for deepfake detection, this type of representation turns out to be relevant: synthetic speech often contains subtle phonetic irregularities and unnatural prosodic patterns that acoustic models may overlook. Together, the two models offer complementary views of the same audio signal, which is the core idea behind our fusion approach.

2.3 Classification Architectures for Spoofing Detection: A Review of the Nes2Net Network \- צריך אולי להרחיב ולפרט יותר על הארכיטקטורה

For the classification part of our system, we decided to base our work on the Nes2Net architecture. Nes2Net is a neural network designed specifically for finding spoofing artifacts in audio signals. It uses a mix of convolutional layers and attention mechanisms to focus on the parts of the audio that matter most. Based on the literature we read, it serves as a strong and reliable baseline classifier, making it a good fit for processing the combined embeddings we extract from WavLM and Whisper.

**3\. Dataset and Preprocessing**

3.1 The Selected Dataset

To train and evaluate our model, we used a combination of two standard anti-spoofing benchmarks: the ASVspoof 2019 Logical Access (LA) dataset and the ASVspoof5 (2024) dataset.

The ASVspoof 2019 LA partition is one of the most widely adopted benchmarks in the speech anti-spoofing field. It contains recordings generated by 19 different text-to-speech and voice-conversion algorithms, alongside genuine human speech samples collected in a controlled recording environment. Using this dataset allows direct comparison with a large body of prior work.

The ASVspoof5 (2024) dataset is the most recent release in the ASVspoof series. It was collected under more challenging and diverse conditions, with a broader range of modern synthesis systems. Including it alongside the 2019 data exposes the model to a wider variety of spoofing artifacts, which supports better generalization.

Both datasets distribute their audio files in FLAC format and provide accompanying protocol files that map each utterance ID to a label (bonafide or spoof) and a source system identifier. Our data organization pipeline reads these protocol files for all available splits — train, dev, and eval for 2019, and train and dev for 2024 — and merges the entries into a single unified pool before re-splitting.

3.2 Audio Signal Preprocessing

Before feature extraction, each audio clip goes through a sequence of normalization steps implemented in the feature extraction pipeline.

**Channel reduction.** If a recording contains more than one channel, the channels are averaged into a single mono waveform. All pre-trained models used in this project expect single-channel input.

**Sample rate normalization.** Every waveform is resampled to 16,000 Hz using torchaudio. This is the expected input rate for both WavLM and the Whisper encoder.

**Fixed-length normalization.** To ensure uniform input dimensions across all samples, each waveform is normalized to exactly 4 seconds (64,000 samples). Clips longer than 4 seconds are randomly cropped: a start offset is drawn uniformly at random from the valid range, and 64,000 consecutive samples are taken from that position. This random crop acts as a mild data augmentation. Clips shorter than 4 seconds are zero-padded at the end to reach the target length.

**Attention masking.** For each clip, we record the ratio of real (non-padded) samples to the total length. This ratio is used to build a frame-level binary mask after feature extraction: frames corresponding to real audio are marked 1, frames corresponding to padding are marked 0. This mask is stored alongside the features and can be used by the classifier to avoid attending to padding artifacts.

**Offline feature extraction.** Rather than extracting features on the fly during training, both the WavLM and Whisper representations are computed once for all clips and saved to disk as PyTorch tensor files (.pt). Each saved file contains the WavLM feature sequence, the Whisper encoder feature sequence (temporally aligned to WavLM), both masks, the integer label (0 for real, 1 for fake), and metadata (source path, split name, class name, sample rate, duration). This offline approach avoids redundant forward passes through the large pre-trained models during training and significantly reduces GPU memory pressure.

**Temporal alignment between models.** WavLM and Whisper produce feature sequences of different temporal lengths for the same input. After extracting both, the Whisper sequence is aligned to the WavLM sequence length: if Whisper produces more frames than WavLM, it is truncated; if it produces fewer frames, it is zero-padded at the end. This alignment is necessary so that the two feature sequences can be processed together by the downstream classifier.

3.3 Data Splitting (Train, Validation, and Test Sets)

After collecting all entries from both datasets and matching each utterance ID against the available audio files, the merged pool is split into train, validation (dev), and test sets using an 80 / 10 / 10 ratio.

Class balance is enforced at the split level. Real and fake entries are shuffled independently (using random seed 42 for reproducibility), and the same number of samples is drawn from each class for every split. This guarantees that no split is skewed toward either class, and that the evaluation metrics reflect genuine discriminative performance rather than class prior.

The actual dataset sizes used in our experiments were as follows:

| Split | Real | Fake | Total |
|-------|------|------|-------|
| Train | 42,139 | 42,139 | 84,278 |
| Dev   | 5,267  | 5,267  | 10,534 |
| Test  | 5,268  | 5,268  | 10,536 |
| **Total** | **52,674** | **52,674** | **105,348** |

The pipeline also produces a protocol file and a summary file for each split, recording each utterance's ID, class label, source dataset (2019\_LA or 2024\_ASVspoof5), and original split, which facilitates traceability and later analysis.

**4\. System Architecture and Proposed Methodology**

4.1 End-to-End System Overview

FusionGuardNet is built around three sequential stages that are cleanly separated by design.

The first stage is offline feature extraction. Before any training takes place, every audio clip in the dataset is passed through two frozen pre-trained encoders — WavLM and Whisper — and their output feature sequences are saved to disk as PyTorch tensor files. This happens once and is not repeated during training. The extraction process and the resulting file format are described in detail in section 3.2.

The second stage is feature fusion. At training and inference time, the two pre-extracted feature sequences for a given clip are loaded from disk and combined by a small learnable module. The output is a single fused feature sequence of the same shape as each individual input.

The third stage is classification. The fused sequence is passed through the Nes2Net backbone, which processes it with a stack of multi-scale residual blocks, reduces the temporal dimension via global pooling, and produces a two-class output (real or fake).

The key architectural choice across all three stages is that the two pre-trained encoders are never fine-tuned. Their weights are frozen throughout. The only components that are trained are the fusion layer and the Nes2Net backbone. This keeps the number of trainable parameters small, prevents catastrophic forgetting of the rich representations learned during large-scale pre-training, and makes it feasible to train the entire system in a small number of epochs.

4.2 Acoustic and Linguistic Feature Representations: WavLM and Whisper

The decision to use two encoders rather than one was motivated by the observation that synthetic speech can fail in more than one way. Some artifacts are acoustic — unnatural spectral texture, phase inconsistencies, or voicing anomalies introduced by the signal-level synthesis process. Others are phonetic or prosodic — unnatural timing between phonemes, irregular rhythm, or inconsistencies in how stress and intonation are distributed across an utterance. A single encoder optimized for one type of information may be blind to the other. WavLM and Whisper were chosen because they represent two distinct perspectives on the same audio signal, trained under fundamentally different objectives.

WavLM (microsoft/wavlm-base-plus) is a self-supervised speech model. It consists of a 12-layer Transformer encoder with 768-dimensional hidden states, 8 attention heads, and approximately 94.7 million parameters. It was pre-trained on around 94,000 hours of speech (Libri-Light, GigaSpeech, and VoxPopuli) using a masked prediction objective on noisy and clean speech pairs, which forces the model to develop a detailed internal representation of acoustic structure. For our task, WavLM's value lies in its sensitivity to low-level signal properties: fine-grained spectral and temporal patterns, voicing characteristics, and the subtle acoustic fingerprints that speech synthesis systems tend to leave behind.

Whisper (openai/whisper-small) is a supervised automatic speech recognition model. Its encoder consists of a two-layer convolutional stem — which processes an 80-bin log-mel spectrogram and downsamples the temporal dimension by a factor of two — followed by 12 Transformer layers with 768-dimensional hidden states and 12 attention heads. It was pre-trained on approximately 680,000 hours of transcribed audio collected from diverse internet sources, with the objective of accurately transcribing spoken content. This ASR-driven training leads Whisper to encode phonetic and prosodic structure: how speech sounds are organized into phonemes, syllables, and words, and how those are distributed over time. Synthetic speech often contains phonetic timing irregularities and prosodic inconsistencies that fall outside WavLM's primary focus.

A practical advantage of this pairing is that both encoders produce feature sequences with the same hidden dimension: 768. No projection layer is needed to make the two representations compatible before fusion. Both also operate on 16 kHz mono audio and produce output sequences at a comparable temporal resolution, which simplifies alignment. The temporal alignment procedure used when both outputs are not identical in length is described in section 3.2.

4.3 Feature Fusion: Channel-wise Learnable Weighted Sum

Given two feature sequences of identical shape — one from WavLM and one from Whisper, each of shape [B, 768, T] — the fusion module must produce a single sequence that carries information from both. We considered several approaches before settling on a channel-wise learnable weighted sum.

Simple concatenation along the channel dimension would produce a 1536-dimensional sequence. This doubles the input size to the backbone and would require either a projection layer to restore the original dimensionality or a backbone with twice the input capacity. Both options add parameters and complexity without a clear benefit.

Cross-attention fusion, where one sequence attends over the other, allows richer cross-modal interaction but introduces significant computational overhead and additional design choices (number of heads, residual structure, positional encoding). Given that both encoders are frozen and the fusion is meant to be a lightweight learned combination, this felt like an unnecessary level of complexity.

We instead use a channel-wise learnable weighted sum, implemented in the `LearnableWeightedSumFusion` module. For each of the 768 channels, the module maintains two learnable scalar parameters: one associated with the WavLM stream and one with the Whisper stream. Before combining, these two scalars are passed through a softmax over the two-source axis, ensuring they sum to 1.0 per channel. The fused value at channel c and time step t is then:

fused[c, t] = w_wavlm[c] · wavlm[c, t] + w_whisper[c] · whisper[c, t]

where w_wavlm[c] + w_whisper[c] = 1 for every channel c.

This design has several practical advantages. The total number of parameters in the fusion layer is 2 × 768 = 1,536 scalars — negligible relative to the backbone. The output remains a 768-dimensional sequence, so the backbone architecture does not need to change based on the fusion approach. And the softmax gating is interpretable: after training, the learned weights reveal which channels the model has come to rely more on WavLM for and which it trusts Whisper on, potentially reflecting a learned division between acoustic and linguistic processing.

4.4 The Classification Model: Adapting Nes2Net to Accept the Fused Features

The Nes2Net backbone receives the fused feature sequence of shape [B, 768, T=200] and produces a two-class prediction.

Its structure is organized around two nested levels of channel decomposition, following the Nes2Net design principle of building hierarchical multi-scale representations without reducing temporal resolution until the final pooling step.

At the outer level, the 768 input channels are divided into 8 groups of 96 channels each. Seven of these groups are processed by their own Bottle2neck block; the eighth passes through unchanged as a residual bypass. The groups are not processed independently: group i receives its own channel slice plus the cumulative output from the previous block before being processed. This cross-group residual accumulation creates nested dependencies across the channel groups and allows later blocks to refine representations built by earlier ones.

Each Bottle2neck block operates on a 96-channel slice and applies an inner Res2Net decomposition. The 96 channels are split into 8 sub-groups of 12. Seven of these sub-groups pass through a dilated 3×3 convolution (dilation=2) in a hierarchical residual chain: sub-group i receives its own slice added to the output of the previous sub-group's convolution before being convolved. This creates a range of effective receptive fields within a single block, allowing it to capture both short-range and longer-range temporal patterns simultaneously. The seventh sub-group's output and the eighth (unchanged) sub-group are concatenated and projected back to 96 channels through a 1×1 convolution. After projection, a Squeeze-and-Excitation module recalibrates channel importance: it computes a global average over the time axis, passes the result through a two-layer bottleneck (96 → 12 → 96) with a sigmoid activation, and multiplies the output back channel-wise. This lets the block selectively amplify channels that carry discriminative information and suppress those that do not. A residual connection adds the block's input to its output.

After the seven Bottle2neck blocks, all eight groups (seven processed, one bypass) are concatenated to reconstruct the full [B, 768, T] representation. A batch normalization and ReLU are applied before temporal aggregation.

The temporal dimension is collapsed by global mean pooling — averaging across all T=200 time steps — producing a single 768-dimensional vector per sample. A dropout layer (p=0.5) is applied for regularization, followed by a linear layer that maps the 768-dimensional vector to 2 output logits, one per class.

The entire backbone was designed to process high-dimensional foundation model features without bottleneck projections, maintaining the full 768-dimensional representation through all intermediate stages. The multi-scale structure (outer nested groups and inner Bottle2neck sub-groups, both with hierarchical residuals) gives the classifier sensitivity to spoofing artifacts at multiple temporal scales, while the SE modules ensure that uninformative channels are down-weighted at each processing step.

**5\. Experimental Setup**

5.1 Environment and Software Libraries 

5.2 Training Procedure (Loss function, optimization algorithm..)

5.3 Hyperparameter Tuning (?)

5.4 Evaluation Metrics (e.g., Equal Error Rate (EER), Accuracy, AUC)

**6\. Results and Discussion**

6.1 Baseline Model Performance (WavLM \+ NES2NET only)

6.2 Enhanced Model Performance (WavLM \+ Whisper \+ NES2NET)

6.3 Ablation Study: Demonstrating the Relative Contribution of the Whisper Integration (?)

6.4 Error Analysis: Strengths and Weaknesses Across Different Spoofing Types

**7\. Conclusion and Future Work**

7.1 Summary of Achievements in Model Integration

7.2 Suggestions for Improvement and Future Research

**8\. References**
