import torch
from torch import nn

### Basic module: Conv2d + BatchNorm + ReLU
class Conv2d_bn(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, strides, padding):
        super().__init__()

        if isinstance(kernel_size, tuple):
            padding_val: int | tuple[int, int] = (
                tuple(k // 2 for k in kernel_size) if padding == "same" else (0, 0)
            )
        else:
            padding_val = kernel_size // 2 if padding == "same" else 0

        self.conv = nn.Conv2d(in_filters, out_filters, 
                              kernel_size = kernel_size, 
                              stride = strides, 
                              padding = padding_val)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()
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
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    

### ResNet Identity Block: (n, n, in_channels) -> (n, n, filters[2])
class ResNetIdentityBlock(nn.Module):

    def __init__(self, in_channels, filters, kernel_size, stride=1):
        super().__init__()
        self.main_path = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = filters[0], kernel_size = 1, strides = stride, padding = "valid"),
            Conv2d_bn(in_filters = filters[0], out_filters = filters[1], kernel_size = kernel_size, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = filters[1], out_filters = filters[2], kernel_size = 1, strides = 1, padding = "valid"),
        )
        self.relu = nn.ReLU()
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
        y = self.relu(self.main_path(x) + x) # Skip connection after a main-path block
        return y

### ResNet Convolutional Block: (2n, 2n, in_channels) -> (n, n, filters[2])
class ResNetConvBlock(nn.Module):

    def __init__(self, in_channels, filters, kernel_size):
        super().__init__()
        self.main_path = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = filters[0], kernel_size = 1, strides = 2, padding = "valid"),
            Conv2d_bn(in_filters = filters[0], out_filters = filters[1], kernel_size = kernel_size, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = filters[1], out_filters = filters[2], kernel_size = 1, strides = 1, padding = "valid"),
        )
        
        self.shortcut_path = nn.Sequential(
            nn.Conv2d(in_channels = in_channels, out_channels = filters[2], kernel_size = 1, stride = 2),
            nn.BatchNorm2d(filters[2]),
        )

        self.relu = nn.ReLU()
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

class ResNet_Light(nn.Module):
    def __init__(self, num_classes = 5):
        super().__init__()
        self.network = nn.Sequential(
            
            # Stage 1: stem
            nn.Conv2d(1, 64, kernel_size = 7, stride = 2), 
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size = 3, stride = 2),

            # Stage 2: Reduced from 2 to 1 Identity Block
            ResNetConvBlock(64, [64, 64, 256], kernel_size = 3),
            nn.Dropout(0.2),
            ResNetIdentityBlock(256, [64, 64, 256], kernel_size = 3),

            # Stage 3: Reduced from 3 to 1 Identity Block
            ResNetConvBlock(256, [128, 128, 512], kernel_size = 3),
            nn.Dropout(0.2),
            ResNetIdentityBlock(512, [128, 128, 512], kernel_size = 3),

            # Stage 4: Reduced from 5 to 1 Identity Block
            ResNetConvBlock(512, [256, 256, 1024], kernel_size = 3),
            nn.Dropout(0.2),
            ResNetIdentityBlock(1024, [256, 256, 1024], kernel_size = 3),

            # Stage 5: Reduced from 2 to 1 Identity Block
            ResNetConvBlock(1024, [512, 512, 2048], kernel_size = 3),
            nn.Dropout(0.2),
            ResNetIdentityBlock(2048, [512, 512, 2048], kernel_size = 3),

            nn.AdaptiveAvgPool2d((1, 1)), 
        )
        self.out = nn.Linear(2048, num_classes)

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