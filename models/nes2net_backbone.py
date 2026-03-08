import torch
import torch.nn as nn
import math


class SEModule(nn.Module):
    def __init__(self, channels, se_ratio=8):
        super().__init__()

        hidden = max(1, channels // se_ratio)

        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, hidden, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(hidden, channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.se(x)
        return x * scale


class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=3, dilation=2, scale=8, se_ratio=8):
        super().__init__()

        width = int(math.floor(planes / scale))
        self.width = width
        self.nums = scale - 1

        self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(width * scale)

        convs = []
        bns = []

        num_pad = math.floor(kernel_size / 2) * dilation

        for _ in range(self.nums):
            convs.append(
                nn.Conv1d(
                    width,
                    width,
                    kernel_size=kernel_size,
                    dilation=dilation,
                    padding=num_pad
                )
            )
            bns.append(nn.BatchNorm1d(width))

        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(bns)

        self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
        self.bn3 = nn.BatchNorm1d(planes)

        self.relu = nn.ReLU()
        self.se = SEModule(planes, se_ratio)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.relu(out)
        out = self.bn1(out)

        spx = torch.split(out, self.width, dim=1)

        for i in range(self.nums):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.convs[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)

            if i == 0:
                out_cat = sp
            else:
                out_cat = torch.cat((out_cat, sp), dim=1)

        out = torch.cat((out_cat, spx[self.nums]), dim=1)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.bn3(out)
        out = self.se(out)

        out = out + residual
        return out


class ASTP(nn.Module):
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super().__init__()

        self.global_context_att = global_context_att

        if global_context_att:
            self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
        else:
            self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)

        self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

    def forward(self, x):
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


class Nes2NetBackbone(nn.Module):
    """
    Backend classifier.
    Expected input shape: [B, C, T]
    """
    def __init__(
        self,
        input_channels,
        nes_ratio=(8, 8),
        dilation=2,
        pool_func="mean",
        se_ratio=(8,),
        num_classes=2
    ):
        super().__init__()

        self.nes_ratio = nes_ratio[0]

        if input_channels % nes_ratio[0] != 0:
            raise ValueError(
                f"input_channels={input_channels} must be divisible by nes_ratio[0]={nes_ratio[0]}"
            )

        c = input_channels // nes_ratio[0]
        self.c = c

        blocks = []
        bns = []

        for _ in range(nes_ratio[0] - 1):
            blocks.append(
                Bottle2neck(
                    c,
                    c,
                    kernel_size=3,
                    dilation=dilation,
                    scale=nes_ratio[1],
                    se_ratio=se_ratio[0]
                )
            )
            bns.append(nn.BatchNorm1d(c))

        self.blocks = nn.ModuleList(blocks)
        self.bns = nn.ModuleList(bns)

        self.bn = nn.BatchNorm1d(input_channels)
        self.relu = nn.ReLU()
        self.pool_func = pool_func

        final_dim = input_channels
        if pool_func == "ASTP":
            final_dim = input_channels * 2
            self.pooling = ASTP(in_dim=input_channels, bottleneck_dim=128)

        self.fc = nn.Linear(final_dim, num_classes)

    def forward(self, x):
        spx = torch.split(x, self.c, dim=1)

        for i in range(self.nes_ratio - 1):
            if i == 0:
                sp = spx[i]
            else:
                sp = sp + spx[i]

            sp = self.blocks[i](sp)
            sp = self.relu(sp)
            sp = self.bns[i](sp)

            if i == 0:
                out = sp
            else:
                out = torch.cat((out, sp), dim=1)

        out = torch.cat((out, spx[-1]), dim=1)
        out = self.bn(out)
        out = self.relu(out)

        if self.pool_func == "mean":
            out = torch.mean(out, dim=-1)
        elif self.pool_func == "ASTP":
            out = self.pooling(out)
        else:
            raise ValueError(f"Unsupported pool_func: {self.pool_func}")

        out = self.fc(out)
        return out