import torch
import torch.nn as nn
from transformers import WavLMModel, WhisperModel
from models.nes2net import Nes2Net # עכשיו זה עובד מול הקובץ החדש

class FusionSystem(nn.Module):
    def __init__(self, mode='fusion', freeze_backbones=True):
        """
        mode: 'wavlm' (768), 'whisper' (768), or 'fusion' (1536)
        """
        super(FusionSystem, self).__init__()
        self.mode = mode
        
        # 1. טעינת Backbone לפי הצורך
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
            
        # הקפאה (כדי לא לאמן את המפלצות)
        if freeze_backbones:
            if self.wavlm:
                for p in self.wavlm.parameters(): p.requires_grad = False
            if self.whisper:
                for p in self.whisper.parameters(): p.requires_grad = False
        
        # 2. טעינת Nes2Net עם הגודל המחושב!
        print(f"Initializing Nes2Net with input_dim={input_dim}...")
        self.backend = Nes2Net(input_channels=input_dim)

    def forward(self, x):
        # x is raw audio [Batch, Length]
        
        features_list = []
        
        # A. חילוץ WavLM
        if self.wavlm:
            # WavLM מחזיר [Batch, Time, 768]
            out = self.wavlm(x).last_hidden_state
            features_list.append(out)
            
        # B. חילוץ Whisper
        if self.whisper:
            # Whisper Encoder מחזיר [Batch, Time, 768]
            # FIX: x is an audio waveform, but Whisper expects log-mel spectrograms. We need to use the preprocessor to procces x before passing it to Whisper.
            out = self.whisper.encoder(x).last_hidden_state 
            features_list.append(out)
            
        # C. חיבור וסנכרון
        if len(features_list) == 2: # Fusion Mode
            wavlm_feat = features_list[0]
            whisper_feat = features_list[1]
            
            # סנכרון אורכים (Whisper ל-WavLM)
            target_len = wavlm_feat.shape[1]
            whisper_feat = whisper_feat.transpose(1, 2)
            whisper_feat = torch.nn.functional.interpolate(whisper_feat, size=target_len)
            whisper_feat = whisper_feat.transpose(1, 2)
            
            # שרשור
            combined = torch.cat((wavlm_feat, whisper_feat), dim=2) # [Batch, Time, 1536]
        else:
            combined = features_list[0] # Single Mode
            
        # D. הכנה ל-Nes2Net (היפוך מימדים ל-[Batch, Channels, Time])
        combined = combined.transpose(1, 2)
        
        # E. סיווג
        return self.backend(combined)
    

"""
CRITICAL FIXES:
1. Whisper Input: Added `WhisperProcessor`. Whisper requires log-mel spectrograms, not raw audio waveforms.
2. Time Alignment: Replaced `interpolate` with direct slicing (`whisper_feat[:, :target_len, :]`). 
   Whisper auto-pads to 30s; interpolation was incorrectly squashing this padded silence into our 4s window. 
   Since both models output at exactly 50Hz, simple slicing perfectly aligns the frames.
3. Architecture: The module is now properly set up for end-to-end inference (Raw Audio -> Features -> Nes2Net).
"""
# import torch
# import torch.nn as nn
# from transformers import WavLMModel, WhisperModel, WhisperProcessor
# from models.nes2net import Nes2Net 

# class FusionSystem(nn.Module):
#     def __init__(self, mode='fusion', freeze_backbones=True):
#         """
#         mode: 'wavlm' (768), 'whisper' (768), or 'fusion' (1536)
#         """
#         super(FusionSystem, self).__init__()
#         self.mode = mode
        
#         self.wavlm = None
#         self.whisper = None
#         self.whisper_processor = None
        
#         input_dim = 0
        
#         if mode in ['wavlm', 'fusion']:
#             print("Loading WavLM...")
#             self.wavlm = WavLMModel.from_pretrained("microsoft/wavlm-base-plus")
#             input_dim += 768
            
#         if mode in ['whisper', 'fusion']:
#             print("Loading Whisper...")
#             self.whisper = WhisperModel.from_pretrained("openai/whisper-small")
#             # Must load processor for Whisper inference
#             self.whisper_processor = WhisperProcessor.from_pretrained("openai/whisper-small")
#             input_dim += 768
            
#         # Freeze backbone parameters to save memory
#         if freeze_backbones:
#             if self.wavlm:
#                 for p in self.wavlm.parameters(): p.requires_grad = False
#             if self.whisper:
#                 for p in self.whisper.parameters(): p.requires_grad = False
        
#         print(f"Initializing Nes2Net with input_dim={input_dim}...")
#         self.backend = Nes2Net(input_channels=input_dim)

#     def forward(self, x, sample_rate=16000):
#         # x expected as 1D or 2D tensor of raw audio
        
#         features_list = []
        
#         # A. WavLM Extraction
#         if self.wavlm:
#             out_wavlm = self.wavlm(x).last_hidden_state
#             features_list.append(out_wavlm)
            
#         # B. Whisper Extraction
#         if self.whisper:
#             # Process raw audio into log-mel spectrograms specifically for Whisper
#             # Note: processor expects CPU numpy array for processing
#             x_np = x.squeeze().cpu().numpy()
#             whisper_inputs = self.whisper_processor(
#                 x_np, 
#                 sampling_rate=sample_rate, 
#                 return_tensors="pt"
#             ).input_features.to(x.device)
            
#             out_whisper = self.whisper.encoder(whisper_inputs).last_hidden_state
#             features_list.append(out_whisper)
            
#         # C. Alignment and Fusion
#         if len(features_list) == 2: 
#             wavlm_feat = features_list[0]
#             whisper_feat = features_list[1]
            
#             # Fix: Slice Whisper features to exactly match WavLM temporal dimension
#             # Do NOT interpolate, as Whisper auto-pads to 30s.
#             target_len = wavlm_feat.shape[1]
#             whisper_feat = whisper_feat[:, :target_len, :]
            
#             # Concatenate along the feature dimension
#             combined = torch.cat((wavlm_feat, whisper_feat), dim=2) 
#         else:
#             combined = features_list[0]
            
#         # D. Permute dimensions for Nes2Net from [Batch, Time, Channels] to [Batch, Channels, Time]
#         combined = combined.transpose(1, 2)
        
#         # E. Final Classification
#         return self.backend(combined)