import torch
import torch.nn as nn

from models.nes2net_backbone import Nes2NetBackbone


class LearnableWeightedSumFusion(nn.Module):
    """
    Channel-wise learnable weighted sum fusion.

    Inputs:
        wavlm_feat   : [B, 768, T]
        whisper_feat : [B, 768, T]

    Output:
        fused_feat   : [B, 768, T]
    """
    def __init__(self, channels=768, use_softmax_gate=True):
        super().__init__()

        self.use_softmax_gate = use_softmax_gate

        self.wav_weight = nn.Parameter(torch.ones(1, channels, 1))
        self.whisper_weight = nn.Parameter(torch.ones(1, channels, 1))

    def forward(self, wavlm_feat, whisper_feat):
        if wavlm_feat.shape != whisper_feat.shape:
            raise ValueError(
                f"Fusion inputs must have the same shape, got {wavlm_feat.shape} and {whisper_feat.shape}"
            )

        if self.use_softmax_gate:
            weights = torch.cat([self.wav_weight, self.whisper_weight], dim=2)  # [1, C, 2]
            weights = torch.softmax(weights, dim=2)
            wav_w = weights[:, :, 0:1]
            whisper_w = weights[:, :, 1:2]
        else:
            wav_w = self.wav_weight
            whisper_w = self.whisper_weight

        return wav_w * wavlm_feat + whisper_w * whisper_feat


class FusionGuardNet(nn.Module):
    """
    Full fusion model:
    - uses frozen pre-extracted WavLM features
    - uses frozen pre-extracted Whisper features
    - learns:
        1. fusion weights
        2. backend classifier
    """
    def __init__(
        self,
        feature_channels=768,
        nes_ratio=(8, 8),
        dilation=2,
        pool_func="mean",
    se_ratio=(8,),
        num_classes=2,
        use_softmax_gate=True
    ):
        super().__init__()

        self.fusion = LearnableWeightedSumFusion(
            channels=feature_channels,
            use_softmax_gate=use_softmax_gate
        )

        self.backbone = Nes2NetBackbone(
            input_channels=feature_channels,
            nes_ratio=nes_ratio,
            dilation=dilation,
            pool_func=pool_func,
            se_ratio=se_ratio,
            num_classes=num_classes
        )

    def forward(self, wavlm_feat, whisper_feat):
        fused = self.fusion(wavlm_feat, whisper_feat)   # [B, 768, T]
        return self.backbone(fused)


class WavLMBranchNet(nn.Module):
    def __init__(
        self,
        feature_channels=768,
        nes_ratio=(8, 8),
        dilation=2,
        pool_func="mean",
        se_ratio=(8,),
        num_classes=2
    ):
        super().__init__()

        self.backbone = Nes2NetBackbone(
            input_channels=feature_channels,
            nes_ratio=nes_ratio,
            dilation=dilation,
            pool_func=pool_func,
            se_ratio=se_ratio,
            num_classes=num_classes
        )

    def forward(self, wavlm_feat):
        return self.backbone(wavlm_feat)


class WhisperBranchNet(nn.Module):
    def __init__(
        self,
        feature_channels=768,
        nes_ratio=(8, 8),
        dilation=2,
        pool_func="mean",
        se_ratio=(8,),
        num_classes=2
    ):
        super().__init__()

        self.backbone = Nes2NetBackbone(
            input_channels=feature_channels,
            nes_ratio=nes_ratio,
            dilation=dilation,
            pool_func=pool_func,
            se_ratio=se_ratio,
            num_classes=num_classes
        )

    def forward(self, whisper_feat):
        return self.backbone(whisper_feat)