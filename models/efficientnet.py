import torch
import torch.nn as nn
import torch.nn.functional as F

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, se_ratio=0.25):
        super(SEBlock, self).__init__()
        reduced_channels = max(1, int(in_channels * se_ratio))
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, reduced_channels, kernel_size=1),
            Swish(),
            nn.Conv2d(reduced_channels, in_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, 
                 expand_ratio, se_ratio, drop_connect_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.drop_connect_rate = drop_connect_rate
        self.use_residual = stride == 1 and in_channels == out_channels
        
        hidden_dim = in_channels * expand_ratio
        layers = []
        
        # Expansion phase
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                Swish()
            ])
        
        # Depthwise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, 
                      (kernel_size-1)//2, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            Swish()
        ])
        
        # Squeeze-and-excitation
        if se_ratio is not None:
            layers.append(SEBlock(hidden_dim, se_ratio))
        
        # Output phase
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])
        
        self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        out = self.block(x)
        if self.use_residual:
            if self.training and self.drop_connect_rate > 0:
                out = self._drop_connect(out)
            out = out + x
        return out
    
    def _drop_connect(self, x):
        keep_prob = 1.0 - self.drop_connect_rate
        mask = torch.empty(x.shape[0], 1, 1, 1, dtype=x.dtype, device=x.device)
        mask.bernoulli_(keep_prob)
        return x / keep_prob * mask


class EfficientNet(nn.Module):
    def __init__(self, num_classes=1):
        super(EfficientNet, self).__init__()
        
        settings = [
            # expand_ratio, channels, repeats, stride, kernel_size, se_ratio
            [1, 16, 1, 1, 3, 0.25],
            [6, 24, 2, 2, 3, 0.25],
            [6, 40, 2, 2, 5, 0.25],
            [6, 80, 3, 2, 3, 0.25],
            [6, 112, 3, 1, 5, 0.25],
            [6, 192, 4, 2, 5, 0.25],
            [6, 320, 1, 1, 3, 0.25]
        ]
        
        # Stem
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            Swish()
        )
        
        # Build blocks
        blocks = []
        in_channels = 32
        total_blocks = sum(setting[2] for setting in settings)
        block_idx = 0
        
        for expand_ratio, channels, repeats, stride, kernel_size, se_ratio in settings:
            for i in range(repeats):
                current_stride = stride if i == 0 else 1
                drop_rate = 0.2 * block_idx / total_blocks
                
                blocks.append(MBConvBlock(
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=current_stride,
                    expand_ratio=expand_ratio,
                    se_ratio=se_ratio,
                    drop_connect_rate=drop_rate
                ))
                in_channels = channels
                block_idx += 1
        
        self.blocks = nn.Sequential(*blocks)
        
        # Head
        self.head = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            Swish()
        )
        
        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x