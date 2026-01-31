"""Nes2Net backend architecture with Res2Net-style blocks and pooling."""

import torch
import torch.nn as nn
import math

# --- Helper modules ---

class SEModule(nn.Module):
    """Squeeze-and-Excitation block for channel-wise recalibration."""
    def __init__(self, channels, SE_ratio=8):
        """Initialize the SE block.

        Args:
            channels: Number of input/output channels.
            SE_ratio: Channel reduction ratio for the bottleneck.
        """
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        """Apply channel-wise attention."""
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    """Res2Net-inspired bottleneck block with SE recalibration."""
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
        """Initialize the bottleneck block.

        Args:
            inplanes: Input channel count.
            planes: Output channel count.
            kernel_size: Kernel size for internal convolutions.
            dilation: Dilation for internal convolutions.
            scale: Number of channel splits.
            SE_ratio: Squeeze-and-Excitation reduction ratio.
        """
        super(Bottle2neck, self).__init__()
        width = int(math.floor(planes / scale))
        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)
        self.nums = scale - 1
        convs = []
        bns = []
        num_pad = math.floor(kernel_size / 2) * dilation
        for i in range(self.nums):
            convs.append(nn.Conv1d(width, width, kernel_size=kernel_size, dilation=dilation, padding=num_pad))
            bns.append(nn.BatchNorm1d(width))
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)
        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU()
        self.width = width
        self.se = SEModule(planes, SE_ratio)

    def forward(self, x):
        """Forward pass through the bottleneck."""
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)
        spx = torch.split(out, self.width, 1)
        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
        out = torch.cat((out, spx[self.nums]), 1)
        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)
        out += residual
        return out

class ASTP(nn.Module):
    """Attentive statistics pooling layer."""
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        """Initialize the pooling layer.

        Args:
            in_dim: Number of input channels.
            bottleneck_dim: Bottleneck channel size for attention layers.
            global_context_att: Whether to append global mean/std to inputs.
        """
        super(ASTP, self).__init__()
        self.global_context_att = global_context_att
        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
        """Compute attentive mean and standard deviation statistics."""
        if self.global_context_att:
            context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
            context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
            x_in = torch.cat((x, context_mean, context_std), dim=1)
        else:
            x_in = x
        alpha = torch.tanh(self.linear1(x_in))
        alpha = torch.softmax(self.linear2(alpha), dim=2)
        mean = torch.sum(alpha * x, dim=2)
        var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
        std = torch.sqrt(var.clamp(min=1e-10))
        return torch.cat([mean, std], dim=1)

# --- Main backend model ---

class Nes2Net(nn.Module):
    """Nes2Net backend for binary real/fake classification."""
    def __init__(self, input_channels, Nes_ratio=[8, 8], dilation=2, pool_func='mean', SE_ratio=[8]):
        """Initialize the network.

        Args:
            input_channels: Number of input channels (e.g., 768 or 1536).
            Nes_ratio: Split ratios for internal Res2Net blocks.
            dilation: Dilation factor for convolution layers.
            pool_func: Pooling function ("mean" or "ASTP").
            SE_ratio: Squeeze-and-Excitation reduction ratios.
        """
        super(Nes2Net, self).__init__()

        # Dynamic calculations based on input size (important for fusion).
        self.Nes_ratio = Nes_ratio[0]

        # Ensure the channel split is valid for the chosen ratio.
        assert input_channels % Nes_ratio[0] == 0, f"Input channels {input_channels} must be divisible by Nes_ratio {Nes_ratio[0]}"

        C = input_channels // Nes_ratio[0]
        self.C = C
        
        Build_in_Res2Nets = []
        bns = []
        for i in range(Nes_ratio[0] - 1):
            Build_in_Res2Nets.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation, scale=Nes_ratio[1], SE_ratio=SE_ratio[0]))
            bns.append(nn.BatchNorm1d(C))
            
        self.Build_in_Res2Nets = nn.ModuleList(Build_in_Res2Nets)
        self.bns = nn.ModuleList(bns)
        
        self.bn = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU()
        self.pool_func = pool_func

        # Final classification head.
        final_dim = input_channels
        if pool_func == 'ASTP':
            final_dim = input_channels * 2  # ASTP doubles dimensions (mean + std).
            self.pooling = ASTP(in_dim=input_channels, bottleneck_dim=128)

        self.fc = nn.Linear(final_dim, 2)  # Binary classes: Real/Fake.

    def forward(self, x):
        """Forward pass.

        Args:
            x: Input tensor shaped [batch, channels, time].

        Returns:
            Logits tensor shaped [batch, 2].
        """
        # Expected input: [Batch, Channels, Time]

        spx = torch.split(x, self.C, 1)
        for i in range(self.Nes_ratio - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]
            sp = self.Build_in_Res2Nets[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)
            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), 1)
                
        out = torch.cat((out, spx[-1]), 1)
        out = self.bn(out)
        out = self.relu(out)
        
        if self.pool_func == 'mean':
            out = torch.mean(out, dim=-1)
        elif self.pool_func == 'ASTP':
            out = self.pooling(out)
            
        out = self.fc(out)
        return out
