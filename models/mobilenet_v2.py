import torch
import torch.nn as nn
import torch.nn.functional as F

class InvertedResidual(nn.Module):
    '''Inverted residual block with expansion'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = expansion * in_planes
        
        self.use_res_connect = self.stride == 1 and in_planes == out_planes
        
        layers = []
        if expansion != 1:
            # Expansion layer
            layers.append(nn.Conv2d(in_planes, hidden_dim, kernel_size=1, stride=1, padding=0, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.ReLU6(inplace=True))
        
        # Depthwise convolution
        layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, 
                               padding=1, groups=hidden_dim, bias=False))
        layers.append(nn.BatchNorm2d(hidden_dim))
        layers.append(nn.ReLU6(inplace=True))
        
        # Pointwise convolution (linear bottleneck)
        layers.append(nn.Conv2d(hidden_dim, out_planes, kernel_size=1, stride=1, padding=0, bias=False))
        layers.append(nn.BatchNorm2d(out_planes))
        
        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1, width_mult=1.0):
        super(MobileNetV2, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        interverted_residual_setting = [
            # expansion, out_channels, num_blocks, stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],  # NOTE: stride 2 for ImageNet
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # Building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        self.features = [nn.Sequential(
            nn.Conv2d(3, input_channel, kernel_size=3, stride=2, padding=1, bias=False),  # NOTE: stride 2 for ImageNet
            nn.BatchNorm2d(input_channel),
            nn.ReLU6(inplace=True)
        )]
        
        # Building inverted residual blocks
        for expansion, out_channels, num_blocks, stride in interverted_residual_setting:
            out_channels = int(out_channels * width_mult)
            layers = []
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(block(input_channel, out_channels, expansion, stride))
                input_channel = out_channels
            self.features.append(nn.Sequential(*layers))
        
        # Building last several layers
        self.features.append(nn.Sequential(
            nn.Conv2d(input_channel, self.last_channel, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(self.last_channel),
            nn.ReLU6(inplace=True)
        ))
        
        self.features = nn.Sequential(*self.features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.last_channel, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
    
