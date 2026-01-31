"""Fusion model that combines WavLM and Whisper backbones with Nes2Net."""

import torch
import torch.nn as nn
from transformers import WavLMModel, WhisperModel
from models.nes2net import Nes2Net

class FusionSystem(nn.Module):
    """Run WavLM, Whisper, or fused features through Nes2Net.

    Args:
        mode: "wavlm", "whisper", or "fusion" to control input channels.
        freeze_backbones: If True, freezes backbone parameters for inference-only use.
    """
    def __init__(self, mode='fusion', freeze_backbones=True):
        """Initialize the fusion system and optional backbones."""
        super(FusionSystem, self).__init__()
        self.mode = mode

        # 1) Load the required backbones.
        self.wavlm = None
        self.whisper = None

        input_dim = 0

        if mode in ['wavlm', 'fusion']:
            print("Loading WavLM...")
            self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
            input_dim += 768

        if mode in ['whisper', 'fusion']:
            print("Loading Whisper...")
            self.whisper = WhisperModel.from_pretrained("openai/whisper-small")
            input_dim += 768

        # Optionally freeze backbone parameters to avoid training them.
        if freeze_backbones:
            if self.wavlm:
                for p in self.wavlm.parameters(): p.requires_grad = False
            if self.whisper:
                for p in self.whisper.parameters(): p.requires_grad = False

        # 2) Initialize Nes2Net with the combined channel size.
        print(f"Initializing Nes2Net with input_dim={input_dim}...")
        self.backend = Nes2Net(input_channels=input_dim)

    def forward(self, x):
        """Run input audio through selected backbones and classify with Nes2Net.

        Args:
            x: Raw audio tensor shaped [batch, length].

        Returns:
            Model logits shaped [batch, 2].
        """
        # x is raw audio [Batch, Length]

        features_list = []

        # A) WavLM feature extraction.
        if self.wavlm:
            # WavLM returns [Batch, Time, 768].
            out = self.wavlm(x).last_hidden_state
            features_list.append(out)

        # B) Whisper feature extraction.
        if self.whisper:
            # Whisper encoder returns [Batch, Time, 768].
            out = self.whisper.encoder(x).last_hidden_state
            features_list.append(out)

        # C) Concatenate features (with length alignment in fusion mode).
        if len(features_list) == 2:  # Fusion Mode
            wavlm_feat = features_list[0]
            whisper_feat = features_list[1]

            # Align Whisper length to match WavLM.
            target_len = wavlm_feat.shape[1]
            whisper_feat = whisper_feat.transpose(1, 2)
            whisper_feat = torch.nn.functional.interpolate(whisper_feat, size=target_len)
            whisper_feat = whisper_feat.transpose(1, 2)

            # Concatenate along feature dimension.
            combined = torch.cat((wavlm_feat, whisper_feat), dim=2) # [Batch, Time, 1536]
        else:
            combined = features_list[0]  # Single Mode

        # D) Prepare for Nes2Net ([Batch, Channels, Time]).
        combined = combined.transpose(1, 2)

        # E) Classify.
        return self.backend(combined)
