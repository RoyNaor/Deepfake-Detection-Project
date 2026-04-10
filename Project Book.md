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

To train and test our model, we use the ASVspoof 2024 and 2019 dataset, which is a standard benchmark in this area of research. It contains a large number of audio clips, split between real human speech and various fake audio generated by text-to-speech and voice conversion algorithms. Using a standard dataset was important for us so we could properly compare our results with existing papers. It also ensures our model is exposed to a wide variety of spoofing techniques during training. (???)

3.2 Audio Signal Preprocessing (Frequency conversion, length truncation, etc.)

3.3 Data Splitting (Train, Validation, and Test Sets)

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
