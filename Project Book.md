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

4.1 Pipeline Overview

4.2 Acoustic and Semantic Feature Extraction: Utilizing WavLM and Whisper

4.3 Feature Fusion: Combining the Outputs of WavLM and Whisper (Maybe we should combine this part with the next one)

4.4 The Classification Model: Adapting Nes2Net to Accept the Fused Features 

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
