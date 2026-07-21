import torch
from torch import nn

### Basic module: Conv2d + BatchNorm + ReLU
class Conv2d_bn(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, strides, padding, activation=True):
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
                              padding = padding_val,
                              bias = False)
        self.bn = nn.BatchNorm2d(out_filters)
        self.relu = nn.ReLU()
        self.activation = activation
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
        if self.activation:
            x = self.relu(x)
        return x

### Inception Stem
class Stem(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        # Input: 128x128 -> Output: (32x32) x 64 channels
        self.block = nn.Sequential(
            Conv2d_bn(in_filters=in_channels, out_filters=32, kernel_size=3, strides=2, padding="same"), # 64x64
            Conv2d_bn(in_filters=32, out_filters=32, kernel_size=3, strides=1, padding="same"),
            Conv2d_bn(in_filters=32, out_filters=64, kernel_size=3, strides=1, padding="same"),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 32x32
        )

    def forward(self, x):
        return self.block(x)

### Inception A Block: identity 32x32 x in_channels -> 32x32 x 128
class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        # Branch 1x1: 32 channels
        self.branch1x1 = Conv2d_bn(in_filters = in_channels, out_filters = 32, kernel_size = 1, strides = 1, padding = "same")

        # Branch 5x5: 16 -> 32 channels
        self.branch5x5 = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 16, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 16, out_filters = 32, kernel_size = 5, strides = 1, padding = "same"),
        )

        # Branch 3x3 double: 16 -> 32 -> 32 channels
        self.branch3x3dbl = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 16, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 16, out_filters = 32, kernel_size = 3, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 32, out_filters = 32, kernel_size = 3, strides = 1, padding = "same"),
        )

        # Branch Pool: AvgPool -> 32 channels
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1),
            Conv2d_bn(in_filters = in_channels, out_filters = 32, kernel_size = 1, strides = 1, padding = "same"),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3dbl = self.branch3x3dbl(x)
        branch_pool = self.branch_pool(x)
        # Concat -> 128 channels
        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], dim=1)

### Reduction Block: 32x32 x in_channels -> 16x16 x 256
class InceptionReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # Branch 3x3: 64 channels
        self.branch3x3 = Conv2d_bn(in_filters = in_channels, out_filters = 64, kernel_size = 3, strides = 2, padding = "same")
        # Branch 3x3 double: 32 -> 64 -> 64 channels
        self.branch3x3dbl = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 32, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 32, out_filters = 64, kernel_size = 3, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 64, out_filters = 64, kernel_size = 3, strides = 2, padding = "same"),
        )
        # Branch Pool: in_channels -> in_channels
        self.branch_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl(x)
        branch_pool = self.branch_pool(x)
        # Concat -> 128 + in_channels
        return torch.cat([branch3x3, branch3x3dbl, branch_pool], dim=1)

### Inception B Block: 16x16 x in_channels -> 16x16 x 256
class InceptionB(nn.Module):
    def __init__(self, in_channels, internal_filters = 64): 
        super().__init__()
        self.branch1x1 = Conv2d_bn(in_filters = in_channels, out_filters = 128, kernel_size = 1, strides = 1, padding = "same")

        self.branch7x7 = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = internal_filters, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = internal_filters, kernel_size = (1, 7), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = 128, kernel_size = (7, 1), strides = 1, padding = "same"),
        )

        self.branch7x7dbl = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = internal_filters, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = internal_filters, kernel_size = (7, 1), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = internal_filters, kernel_size = (1, 7), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = internal_filters, kernel_size = (7, 1), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = 128, kernel_size = (1, 7), strides = 1, padding = "same"),
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1),
            Conv2d_bn(in_filters = in_channels, out_filters = 128, kernel_size = 1, strides = 1, padding = "same"),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7dbl = self.branch7x7dbl(x)
        branch_pool = self.branch_pool(x)
        # Concat -> 128 + 128 + 128 + 128 = 512 channels
        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], dim=1)

    
class MinInception(nn.Module):
    def __init__(self, in_channels=1, out_features=5):
        super().__init__()

        self.stem = Stem(in_channels)
        
        self.inception_a = nn.Sequential(
            InceptionA(in_channels=64),
            InceptionA(in_channels=128),
        )
        
        self.reduction_a = InceptionReductionA(in_channels=128) 
        
        self.inception_b = InceptionB(in_channels=256, internal_filters=64)

        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.2) 
        self.out = nn.Linear(512, out_features) 

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        y = self.out(x)
        
        return y