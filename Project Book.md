**1\. Introduction**

1.1. Challenges in Audio-Based Deepfake Detection

The rapid proliferation of neural speech synthesis and voice conversion (VC) systems has fundamentally altered the threat landscape for audio authentication. Early synthetic speech systems—based on concatenative or parametric statistical methods—introduced audible artifacts that made detection relatively tractable using classical signal processing approaches. However, modern generative architectures have dramatically closed this gap. GAN-based vocoders such as HiFi-GAN and MelGAN, normalizing flow models such as Glow-TTS, transformer-based TTS systems such as FastSpeech 2 and VITS, and diffusion-based synthesis models such as DiffWave and Grad-TTS can now produce speech that is perceptually indistinguishable from genuine human recordings [Wang2023, Kumar2019]. Zero-shot voice cloning systems like VALL-E and YourTTS further lower the barrier by requiring only a few seconds of target speaker audio to synthesize a convincing impersonation [Wang2023VALLE, Casanova2022]. The social and security implications are severe: synthesized speech threatens the integrity of Automatic Speaker Verification (ASV) systems deployed in banking, access control, and forensic applications, while also enabling disinformation campaigns, social engineering attacks, and financial fraud at scale [Kinnunen2020].

The research community has tracked this evolving threat through the ASVspoof benchmark series. ASVspoof 2015 addressed classical TTS and VC systems. ASVspoof 2017 introduced replay attacks. ASVspoof 2019—the primary benchmark used in this work—defined the Logical Access (LA) track covering 19 distinct TTS and VC spoofing algorithms, including WaveNet, SampleRNN, and neural vocoders, across a corpus of over 100,000 utterances [Todisco2019]. The evaluation set spans both seen (A01–A06) and unseen (A07–A19) spoofing systems, specifically designed to test generalization. The standard CQCC-GMM and LFCC-GMM baselines achieved Equal Error Rates (EER) of 9.57% and 8.09% on this evaluation set, respectively, revealing the limitations of classical approaches. ASVspoof 2021 extended evaluation to in-the-wild conditions with audio transmitted through telephone codecs (G.711, G.722) and VoIP channels; many systems that achieved sub-1% EER on clean recordings degraded substantially under these conditions [Yi2022], highlighting the real-world fragility of lab-optimized detectors.

A core reason classical front-end features fail is their fundamental design objective: Mel-Frequency Cepstral Coefficients (MFCC), Linear Frequency Cepstral Coefficients (LFCC), and Constant-Q Cepstral Coefficients (CQCC) were engineered for speech recognition and speaker identification, not artifact detection. These representations capture coarse spectral envelope shape but discard fine-grained spectral discontinuities, phase artifacts, and temporal micro-irregularities that neural vocoders introduce at the waveform level. Additionally, their fixed filter-bank structures prevent them from adapting to the diverse and evolving artifact signatures produced by different synthesis paradigms. This structural brittleness is directly visible in ASVspoof evaluations: per-system EERs vary by orders of magnitude, and systems that handle seen attack types well often fail dramatically on unseen ones [Todisco2019, Muller2022].

This generalization gap—between known and unknown attacks—is arguably the most pressing open challenge in anti-spoofing. In practice, defenders cannot anticipate every TTS system an adversary might deploy. The open-source ecosystem of voice synthesis tools (Coqui TTS, Tortoise TTS, OpenVoice) continues to grow rapidly, making it unrealistic to enumerate all possible spoofing systems at training time. Cross-dataset generalization compounds the problem further: models trained on ASVspoof 2019 have been shown to degrade substantially when evaluated on independent corpora with different recording conditions, languages, and synthesis pipelines [Muller2022]. Adversarial attacks represent an additional vector—targeted perturbations can cause state-of-the-art countermeasures to misclassify synthetic speech as genuine with high confidence, even when the perturbation is imperceptible to human listeners [Zhang2022].

A major paradigm shift came with the application of Self-Supervised Learning (SSL) models to anti-spoofing. Models such as wav2vec 2.0 [Baevski2020], HuBERT [Hsu2021], and WavLM [Chen2022], pre-trained on tens of thousands of hours of unlabeled speech, learn rich representations that generalize far better to unseen attacks than handcrafted features. WavLM, trained on 94,000 hours of diverse speech, captures fine-grained signal-level artifacts through its masked speech prediction objective on noisy inputs. Wang et al. (2021) demonstrated that wav2vec 2.0 features combined with a lightweight backend achieve highly competitive EER. The AASIST system [Jung2022], combining SSL features with a graph attention network, reported 0.83% EER on ASVspoof 2019 LA—more than a 10× improvement over the LFCC-GMM baseline. The Nes2Net architecture [Liu2024], which serves as the classification backend in our system, further advances efficiency and performance by incorporating nested residual connections and Squeeze-and-Excitation attention over SSL-derived features.

However, SSL acoustic models primarily capture signal-level and spectro-temporal artifacts. A complementary category of artifacts exists at the semantic and phonetic level: unnatural coarticulation transitions, irregular prosodic rhythm, anomalous stress patterns, and phoneme-level inconsistencies that result from autoregressive generation errors or imperfect prosody transfer. Whisper [Radford2023], trained on 680,000 hours of transcribed speech specifically for Automatic Speech Recognition, encodes precisely this higher-level phonetic and linguistic structure. The hypothesis motivating our work is that combining an acoustic SSL model (WavLM) with an ASR-oriented model (Whisper) provides a more complete and complementary view of the input signal—one that detects both low-level synthesis artifacts and high-level semantic anomalies. No single representation is sufficient; the challenge demands multi-modal fusion. This is the central problem our system, FusionGuardNet, is designed to address.

---

**References for this section:**

- [Todisco2019] M. Todisco, X. Wang, V. Vestman, et al., "ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection," Interspeech, 2019.
- [Yi2022] J. Yi, X. Fu, J. Tao, et al., "ASVspoof 2021: Towards Spoofed and Deepfake Speech Detection in the Wild," IEEE/ACM TASLP, 2022.
- [Muller2022] N. M. Müller, P. Czempin, F. Diekmann, et al., "Does Audio Deepfake Detection Generalize?" Interspeech, 2022.
- [Baevski2020] A. Baevski, Y. Zhou, A. Mohamed, M. Auli, "wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations," NeurIPS, 2020.
- [Hsu2021] W.-N. Hsu, B. Bolte, Y.-H. H. Tsai, et al., "HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units," IEEE/ACM TASLP, 2021.
- [Chen2022] S. Chen, C. Wang, Z. Chen, et al., "WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing," IEEE Journal of Selected Topics in Signal Processing, 2022.
- [Jung2022] J. Jung, H.-S. Heo, H. Tak, et al., "AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks," ICASSP, 2022.
- [Liu2024] H. Liu, et al., "Nes2Net: A Lightweight Nested Architecture for Foundation Model Driven Speech Anti-spoofing," 2024.
- [Radford2023] A. Radford, J. W. Kim, T. Xu, et al., "Robust Speech Recognition via Large-Scale Weak Supervision," ICML, 2023.
- [Kinnunen2020] T. Kinnunen, H. Delgado, N. Evans, et al., "Tandem Assessment of Spoofing Countermeasures and Automatic Speaker Verification: Fundamentals," IEEE/ACM TASLP, 2020.
- [Zhang2022] Z. Zhang, et al., "Adversarial Attacks on Audio Anti-Spoofing Models," 2022.
- [Wang2023VALLE] C. Wang, S. Chen, Y. Wu, et al., "Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers," ICLR, 2023.
- [Casanova2022] E. Casanova, J. Weber, C. D. Shulby, et al., "YourTTS: Towards Zero-Shot Multi-Speaker TTS and Zero-Shot Voice Conversion for Everyone," ICML, 2022.

---

1.2. Project Goals: Improving Spoofing Classification Using Pre-trained Models

The main goal of our project is to improve how accurately a system can tell if an audio file is real or fake. Instead of training a feature extractor from scratch, we decided to use large pre-trained models that already understand audio well. We want to see if combining different types of representations from these models can help the classifier catch subtle mistakes in the fake audio. Ultimately, we aim to deliver a working pipeline that achieves better detection results than standard baseline models.

1.3. Our Contribution: Integrating Whisper with WavLM in the Nes2Net Architecture

Our main addition in this project is integrating the Whisper model alongside WavLM into the Nes2Net network. Normally, WavLM is used to look at the low-level acoustic details of the voice. We hypothesized that by adding Whisper, which is designed for speech recognition, we could also capture higher-level phonetic and semantic features. By combining the outputs from both models, we give the Nes2Net classifier more information to work with, which we demonstrate improves its overall ability to detect spoofed audio.

**2\. Related Work** 

2.1 Classical and Modern Approaches to Audio Spoofing Detection (?)

2.2 Pre-trained Speech and ASR Models: Capabilities and Applications of WavLM and Whisper \- לוודא פרטים

We chose to work with WavLM and Whisper because they are trained differently and offer distinct features. WavLM is a self-supervised model trained on a massive amount of raw audio, making it very good at recognizing speaker characteristics and background environments. Whisper, on the other hand, was trained on transcribed speech, making it strong at understanding phonetics and language content. By researching what these models are typically used for, we realized they could complement each other well for deepfake detection.

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
