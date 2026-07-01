import torch
from torch import nn

class MainPath(nn.Module):

    def __init__(self, in_channels, filters, kernel_size, stride=1):
        super().__init__()
        F1, F2, F3 = filters
        self.main_path = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = F1, kernel_size = 1, stride = stride),
            nn.BatchNorm2d(F1),
            nn.ReLU(),
            nn.Conv2d(in_channels = F1, out_channels = F2, kernel_size = kernel_size, padding = kernel_size // 2),
            nn.BatchNorm2d(F2),
            nn.ReLU(),
            nn.Conv2d(in_channels = F2, out_channels = F3, kernel_size = 1),
            nn.BatchNorm2d(F3),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        y = self.main_path(x)
        return y


class IdentityBlock(MainPath):

    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.relu(self.main_path(x) + x) # Skip connection after a main-path block
        return y


class ConvolutionalBlock(MainPath):

    def __init__(self, in_channels, filters, kernel_size):
        super().__init__(in_channels, filters, kernel_size, stride=2)
        self.relu = nn.ReLU()
        self.shortcut_path = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = filters[2], kernel_size = 1, stride = 2 ),
            nn.BatchNorm2d(filters[2]),
        )
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, x):
        y = self.relu(self.main_path(x) + self.shortcut_path(x))
        return y

class ResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            
            # Stage 1: stem
            nn.Conv2d(3, 64, kernel_size = 7, stride = 2),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            # Stage 2
            ConvolutionalBlock(64, [64, 64, 256], kernel_size = 3),
            nn.Dropout(0.2),
            IdentityBlock(256, [64, 64, 256], kernel_size = 3),
            IdentityBlock(256, [64, 64, 256], kernel_size = 3),

            # Stage 3
            ConvolutionalBlock(256, [128, 128, 512], kernel_size = 3),
            nn.Dropout(0.2),
            IdentityBlock(512, [128, 128, 512], kernel_size = 3),
            IdentityBlock(512, [128, 128, 512], kernel_size = 3),
            IdentityBlock(512, [128, 128, 512], kernel_size = 3),

            # Stage 4
            ConvolutionalBlock(512, [256, 256, 1024], kernel_size = 3),
            nn.Dropout(0.2),
            IdentityBlock(1024, [256, 256, 1024], kernel_size = 3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size = 3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size = 3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size = 3),
            IdentityBlock(1024, [256, 256, 1024], kernel_size = 3),

            # Stage 5
            ConvolutionalBlock(1024, [512, 512, 2048], kernel_size = 3),
            nn.Dropout(0.2),
            IdentityBlock(2048, [512, 512, 2048], kernel_size = 3),
            IdentityBlock(2048, [512, 512, 2048], kernel_size = 3),

            nn.AvgPool2d(kernel_size = 2, stride=2),

        )
        self.classification_layer = nn.Linear(2048, 1)
        self.apply(self._init_weights)

    def forward(self, x):
        y = self.network(x)

        # Flatten for the last layer
        y = y.reshape(x.shape[0], -1) 

        # Last classification layer
        y = self.classification_layer(y)
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
