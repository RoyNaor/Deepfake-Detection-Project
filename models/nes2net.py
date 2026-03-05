import torch
import torch.nn as nn
import math

# --- מחלקות עזר (בדיוק כמו במקור) ---

class SEModule(nn.Module):
    def __init__(self, channels, SE_ratio=8):
        super(SEModule, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
            nn.Sigmoid(),
        )

    def forward(self, input):
        x = self.se(input)
        return input * x

class Bottle2neck(nn.Module):
    def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
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
    def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
        super(ASTP, self).__init__()
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

# --- המודל הראשי (זה ה-Backend שלנו) ---
# שיניתי את השם ל-Nes2Net וסידרתי את ה-input_channels

class Nes2Net(nn.Module):
    def __init__(self, input_channels, Nes_ratio=[8, 8], dilation=2, pool_func='mean', SE_ratio=[8]):
        super(Nes2Net, self).__init__()
        
        # חישובים דינמיים לפי גודל הקלט (חשוב ל-Fusion!)
        self.Nes_ratio = Nes_ratio[0]
        
        # וידוא שהחלוקה עובדת (למשל 1536 / 8 = 192, זה עובד)
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
        
        # שכבת הסיווג הסופית
        final_dim = input_channels
        if pool_func == 'ASTP':
            final_dim = input_channels * 2 # ASTP מכפיל את המימדים (mean + std)
            self.pooling = ASTP(in_dim=input_channels, bottleneck_dim=128)
            
        self.fc = nn.Linear(final_dim, 2) # שיניתי ל-2 מחלקות: Real/Fake

    def forward(self, x):
        # הציפייה לקלט: [Batch, Channels, Time]
        
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
    



"""
=============================================================================
NES2NET-X IMPLEMENTATION UPDATE (Based on Official GitHub):

1. Bottle2neck (Inner Layer) upgraded to Nes2Net-X:
   Instead of using standard addition, this layer now implements the learnable 
   weighted sum mechanism. It stacks the cascading splits along a new 4th 
   dimension (creating a 4D tensor), applies Conv2d, multiplies by explicit 
   learnable parameters (`self.weighted_sum`), and sums them back to 3D.
2. Nes2Net (Outer Layer) macro-architecture:
   Kept the standard addition for the macro-splits, as this perfectly matches 
   the official Nes2Net-X repository logic.
3. Fusion Compatibility:
   Maintained the dynamic channel calculation (input_channels // Nes_ratio[0]) 
   and the 2-class output (fc layer) to seamlessly support both individual 
   models (768) and the concatenated Fusion pipeline (1536).
=============================================================================
"""

# import torch
# import torch.nn as nn
# import math

# class SEModule(nn.Module):
#     def __init__(self, channels, SE_ratio=8):
#         super(SEModule, self).__init__()
#         self.se = nn.Sequential(
#             nn.AdaptiveAvgPool1d(1),
#             nn.Conv1d(channels, channels // SE_ratio, kernel_size=1, padding=0),
#             nn.ReLU(),
#             nn.Conv1d(channels // SE_ratio, channels, kernel_size=1, padding=0),
#             nn.Sigmoid(),
#         )

#     def forward(self, input):
#         x = self.se(input)
#         return input * x


# class Bottle2neck(nn.Module):
#     def __init__(self, inplanes, planes, kernel_size=None, dilation=None, scale=8, SE_ratio=8):
#         super(Bottle2neck, self).__init__()
#         width = int(math.floor(planes / scale))
#         self.conv1 = nn.Conv1d(inplanes, width * scale, kernel_size=1)
#         self.bn1 = nn.BatchNorm1d(width * scale)
#         self.nums = scale - 1
        
#         convs = []
#         bns = []
#         weighted_sum = []
#         num_pad = math.floor(kernel_size / 2) * dilation
        
#         for i in range(self.nums):
#             # Using Conv2d to process stacked 1D channels over the new 4th dimension
#             convs.append(
#                 nn.Conv2d(width, width, kernel_size=(kernel_size, 1), dilation=(dilation, 1), padding=(num_pad, 0))
#             )
#             bns.append(nn.BatchNorm2d(width))
            
#             # Learnable weights for the weighted sum fusion
#             initial_value = torch.ones(1, 1, 1, i + 2) * (1 / (i + 2))
#             weighted_sum.append(nn.Parameter(initial_value, requires_grad=True))
            
#         self.weighted_sum = nn.ParameterList(weighted_sum)
#         self.convs = nn.ModuleList(convs)
#         self.bns = nn.ModuleList(bns)
        
#         self.conv3 = nn.Conv1d(width * scale, planes, kernel_size=1)
#         self.bn3 = nn.BatchNorm1d(planes)
#         self.relu = nn.ReLU()
#         self.width = width
#         self.se = SEModule(planes, SE_ratio)

#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.relu(out)
        
#         # Add a 4th dimension for stacking splits
#         out = self.bn1(out).unsqueeze(-1)  

#         spx = torch.split(out, self.width, 1)
#         sp = spx[self.nums]
        
#         for i in range(self.nums):
#             # Concatenate current split along the new 4th dimension
#             sp = torch.cat((sp, spx[i]), -1)

#             # Apply Conv2d, ReLU, and BatchNorm
#             sp_conv = self.bns[i](self.relu(self.convs[i](sp)))
            
#             # Apply learnable weighted sum
#             sp_s = sp_conv * self.weighted_sum[i]
#             sp_s = torch.sum(sp_s, dim=-1, keepdim=False)

#             if i == 0:
#                 out_fused = sp_s
#             else:
#                 out_fused = torch.cat((out_fused, sp_s), 1)
                
#         # Append the last un-convolved split
#         out_fused = torch.cat((out_fused, spx[self.nums].squeeze(-1)), 1)
        
#         # Final projections
#         out_fused = self.conv3(out_fused)
#         out_fused = self.relu(out_fused)
#         out_fused = self.bn3(out_fused)
#         out_fused = self.se(out_fused)
        
#         # Residual connection
#         out_fused += residual
#         return out_fused


# class ASTP(nn.Module):
#     def __init__(self, in_dim, bottleneck_dim=128, global_context_att=False):
#         super(ASTP, self).__init__()
#         self.global_context_att = global_context_att
#         if global_context_att:
#             self.linear1 = nn.Conv1d(in_dim * 3, bottleneck_dim, kernel_size=1)
#         else:
#             self.linear1 = nn.Conv1d(in_dim, bottleneck_dim, kernel_size=1)
#         self.linear2 = nn.Conv1d(bottleneck_dim, in_dim, kernel_size=1)

#     def forward(self, x):
#         if self.global_context_att:
#             context_mean = torch.mean(x, dim=-1, keepdim=True).expand_as(x)
#             context_std = torch.sqrt(torch.var(x, dim=-1, keepdim=True) + 1e-10).expand_as(x)
#             x_in = torch.cat((x, context_mean, context_std), dim=1)
#         else:
#             x_in = x
            
#         alpha = torch.tanh(self.linear1(x_in))
#         alpha = torch.softmax(self.linear2(alpha), dim=2)
#         mean = torch.sum(alpha * x, dim=2)
#         var = torch.sum(alpha * (x ** 2), dim=2) - mean ** 2
#         std = torch.sqrt(var.clamp(min=1e-10))
#         return torch.cat([mean, std], dim=1)


# class Nes2Net(nn.Module):
#     def __init__(self, input_channels, Nes_ratio=[8, 8], dilation=2, pool_func='mean', SE_ratio=[8]):
#         super(Nes2Net, self).__init__()
        
#         self.Nes_ratio = Nes_ratio[0]
#         assert input_channels % Nes_ratio[0] == 0, f"Input channels {input_channels} must be divisible by {Nes_ratio[0]}"
        
#         C = input_channels // Nes_ratio[0]
#         self.C = C
        
#         Build_in_Res2Nets = []
#         bns = []
        
#         for i in range(Nes_ratio[0] - 1):
#             Build_in_Res2Nets.append(Bottle2neck(C, C, kernel_size=3, dilation=dilation, scale=Nes_ratio[1], SE_ratio=SE_ratio[0]))
#             bns.append(nn.BatchNorm1d(C))
            
#         self.Build_in_Res2Nets = nn.ModuleList(Build_in_Res2Nets)
#         self.bns = nn.ModuleList(bns)
        
#         self.bn = nn.BatchNorm1d(input_channels)
#         self.relu = nn.ReLU()
#         self.pool_func = pool_func
        
#         final_dim = input_channels
#         if pool_func == 'ASTP':
#             final_dim = input_channels * 2 
#             self.pooling = ASTP(in_dim=input_channels, bottleneck_dim=128)
            
#         # Using 2 classes (Bonafide vs Spoof) 
#         self.fc = nn.Linear(final_dim, 2) 

#     def forward(self, x):
#         spx = torch.split(x, self.C, 1)
        
#         for i in range(self.Nes_ratio - 1):
#             if i == 0:
#                 sp = spx[i]
#             else:
#                 # Outer layer remains standard addition as per official implementation
#                 sp = sp + spx[i]
                
#             sp = self.Build_in_Res2Nets[i](sp)
#             sp = self.relu(sp)
#             sp = self.bns[i](sp)
            
#             if i == 0:
#                 out = sp
#             else:
#                 out = torch.cat((out, sp), 1)
                
#         out = torch.cat((out, spx[-1]), 1)
#         out = self.bn(out)
#         out = self.relu(out)
        
#         if self.pool_func == 'mean':
#             out = torch.mean(out, dim=-1)
#         elif self.pool_func == 'ASTP':
#             out = self.pooling(out)
            
#         out = self.fc(out)
#         return out