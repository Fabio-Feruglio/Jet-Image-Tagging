import torch
from torch import nn

from model_blocks import ResNetConvBlock, ResNetIdentityBlock

class ResNet50(nn.Module):

    def __init__(self, num_classes = 5):
        super().__init__()
        self.network = nn.Sequential(
            
            # Stage 1: stem
            nn.Conv2d(1, 64, kernel_size = 7, stride = 2), # adapted to get 1 channel input
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            # Stage 2
            ResNetConvBlock(64, [64, 64, 256], kernel_size = 3),
            nn.Dropout(0.2),
            ResNetIdentityBlock(256, [64, 64, 256], kernel_size = 3),
            ResNetIdentityBlock(256, [64, 64, 256], kernel_size = 3),

            # Stage 3
            ResNetConvBlock(256, [128, 128, 512], kernel_size = 3),
            nn.Dropout(0.2),
            ResNetIdentityBlock(512, [128, 128, 512], kernel_size = 3),
            ResNetIdentityBlock(512, [128, 128, 512], kernel_size = 3),
            ResNetIdentityBlock(512, [128, 128, 512], kernel_size = 3),

            # Stage 4
            ResNetConvBlock(512, [256, 256, 1024], kernel_size = 3),
            nn.Dropout(0.2),
            ResNetIdentityBlock(1024, [256, 256, 1024], kernel_size = 3),
            ResNetIdentityBlock(1024, [256, 256, 1024], kernel_size = 3),
            ResNetIdentityBlock(1024, [256, 256, 1024], kernel_size = 3),
            ResNetIdentityBlock(1024, [256, 256, 1024], kernel_size = 3),
            ResNetIdentityBlock(1024, [256, 256, 1024], kernel_size = 3),

            # Stage 5
            ResNetConvBlock(1024, [512, 512, 2048], kernel_size = 3),
            nn.Dropout(0.2),
            ResNetIdentityBlock(2048, [512, 512, 2048], kernel_size = 3),
            ResNetIdentityBlock(2048, [512, 512, 2048], kernel_size = 3),

            nn.AdaptiveAvgPool2d((1, 1)), # Global Average Pooling
        )
        self.out: nn.Module = nn.Linear(2048, num_classes) # adapted to get 5 classes output
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.network(x)
        y = y.reshape(x.shape[0], -1) 
        y = self.out(y)
        return y

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
