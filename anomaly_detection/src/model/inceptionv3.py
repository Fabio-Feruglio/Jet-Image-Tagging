import torch
from torch import nn

### Basic module: Conv2d + BatchNorm + ReLU
class Conv2d_bn(nn.Module):
    def __init__(self, in_filters, out_filters, kernel_size, strides, padding):
        super().__init__()

        if isinstance(kernel_size, tuple):
            padding_val = (
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

### Inception Stem V3
class InceptionStemV3(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()

        self.block = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 32, kernel_size = 3, strides = 2, padding = "valid"),
            Conv2d_bn(in_filters = 32, out_filters = 32, kernel_size = 3, strides = 1, padding = "valid"),
            Conv2d_bn(in_filters = 32, out_filters = 64, kernel_size = 3, strides = 1, padding = "same"),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0),

            Conv2d_bn(in_filters = 64, out_filters = 80, kernel_size = 1, strides = 1, padding = "valid"),
            Conv2d_bn(in_filters = 80, out_filters = 192, kernel_size = 3, strides = 1, padding = "valid"),
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)
        )

    def forward(self, x):
        return self.block(x)

### Inception A Block
class InceptionA(nn.Module):
    def __init__(self, in_channels, pool_features):
        super().__init__()
        self.branch1x1 = Conv2d_bn(in_filters = in_channels, out_filters = 64, kernel_size = 1, strides = 1, padding = "same")

        self.branch5x5 = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 48, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 48, out_filters = 64, kernel_size = 5, strides = 1, padding = "same"),
        )

        self.branch3x3dbl = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 64, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 64, out_filters = 96, kernel_size = 3, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 96, out_filters = 96, kernel_size = 3, strides = 1, padding = "same"),
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1),
            Conv2d_bn(in_filters = in_channels, out_filters = pool_features, kernel_size = 1, strides = 1, padding = "same"),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch5x5 = self.branch5x5(x)
        branch3x3dbl = self.branch3x3dbl(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch1x1, branch5x5, branch3x3dbl, branch_pool], dim=1)

### Reduction A Block
class InceptionReductionA(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = Conv2d_bn(in_filters = in_channels, out_filters = 384, kernel_size = 3, strides = 2, padding = "valid")

        self.branch3x3dbl = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 64, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 64, out_filters = 96, kernel_size = 3, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 96, out_filters = 96, kernel_size = 3, strides = 2, padding = "valid"),
        )

        self.branch_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch3x3dbl = self.branch3x3dbl(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch3x3, branch3x3dbl, branch_pool], dim=1)

### Inception B Block
class InceptionB(nn.Module):
    def __init__(self, in_channels, internal_filters): 
        super().__init__()
        self.branch1x1 = Conv2d_bn(in_filters = in_channels, out_filters = 192, kernel_size = 1, strides = 1, padding = "same")

        self.branch7x7 = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = internal_filters, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = internal_filters, kernel_size = (1, 7), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = 192, kernel_size = (7, 1), strides = 1, padding = "same"),
        )

        self.branch7x7dbl = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = internal_filters, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = internal_filters, kernel_size = (7, 1), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = internal_filters, kernel_size = (1, 7), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = internal_filters, kernel_size = (7, 1), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = internal_filters, out_filters = 192, kernel_size = (1, 7), strides = 1, padding = "same"),
        )

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1),
            Conv2d_bn(in_filters = in_channels, out_filters = 192, kernel_size = 1, strides = 1, padding = "same"),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch7x7 = self.branch7x7(x)
        branch7x7dbl = self.branch7x7dbl(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch1x1, branch7x7, branch7x7dbl, branch_pool], dim=1)

### Reduction B Block: (35, 35, 768) -> (17, 17, 1280)
class InceptionReductionB(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch3x3 = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 192, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 192, out_filters = 320, kernel_size = 3, strides = 2, padding = "valid"),
        )

        self.branch7x7x3 = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 192, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 192, out_filters = 192, kernel_size = (1, 7), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 192, out_filters = 192, kernel_size = (7, 1), strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 192, out_filters = 192, kernel_size = 3, strides = 2, padding = "valid"),
        )

        self.branch_pool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 0)

    def forward(self, x):
        branch3x3 = self.branch3x3(x)
        branch7x7x3 = self.branch7x7x3(x)
        branch_pool = self.branch_pool(x)
        return torch.cat([branch3x3, branch7x7x3, branch_pool], dim = 1)

### Inception C Block
class InceptionC(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.branch1x1 = Conv2d_bn(in_filters = in_channels, out_filters = 320, kernel_size = 1, strides = 1, padding = "same")

        self.branch3x3 = Conv2d_bn(in_filters = in_channels, out_filters = 384, kernel_size = 1, strides = 1, padding = "same")
        self.branch3x3_1 = Conv2d_bn(in_filters = 384, out_filters = 384, kernel_size = (1, 3), strides = 1, padding = "same")
        self.branch3x3_2 = Conv2d_bn(in_filters = 384, out_filters = 384, kernel_size = (3, 1), strides = 1, padding = "same")

        self.branch3x3dbl = nn.Sequential(
            Conv2d_bn(in_filters = in_channels, out_filters = 448, kernel_size = 1, strides = 1, padding = "same"),
            Conv2d_bn(in_filters = 448, out_filters = 384, kernel_size = 3, strides = 1, padding = "same"),
        )
        self.branch3x3dbl_1 = Conv2d_bn(in_filters = 384, out_filters = 384, kernel_size = (1, 3), strides = 1, padding = "same")
        self.branch3x3dbl_2 = Conv2d_bn(in_filters = 384, out_filters = 384, kernel_size = (3, 1), strides = 1, padding = "same")

        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(kernel_size = 3, stride = 1, padding = 1),
            Conv2d_bn(in_filters = in_channels, out_filters = 192, kernel_size = 1, strides = 1, padding = "same"),
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch3x3 = self.branch3x3(x)
        branch3x3 = torch.cat([self.branch3x3_1(branch3x3), self.branch3x3_2(branch3x3)], dim = 1)

        branch3x3dbl = self.branch3x3dbl(x)
        branch3x3dbl = torch.cat([self.branch3x3dbl_1(branch3x3dbl), self.branch3x3dbl_2(branch3x3dbl)], dim=1)

        branch_pool = self.branch_pool(x)
        
        return torch.cat([branch1x1, branch3x3, branch3x3dbl, branch_pool], dim=1)

### Main Inception V3
class InceptionV3(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()

        self.stem = InceptionStemV3(in_channels)
        
        self.inception_a = nn.Sequential(
            InceptionA(in_channels=192, pool_features=32),
            InceptionA(in_channels=256, pool_features=64),
            InceptionA(in_channels=288, pool_features=64), 
        )
        
        self.reduction_a = InceptionReductionA(in_channels=288) 
        
        self.inception_b = nn.Sequential(
            InceptionB(in_channels=768, internal_filters=128),
            InceptionB(in_channels=768, internal_filters=160),
            InceptionB(in_channels=768, internal_filters=160),
            InceptionB(in_channels=768, internal_filters=192),
        )
        
        self.reduction_b = InceptionReductionB(in_channels=768)
        
        self.inception_c = nn.Sequential(
            InceptionC(in_channels=1280),
            InceptionC(in_channels=2048),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.5) 
        self.out: nn.Module = nn.Linear(2048, num_classes) 

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        y = self.out(x)
        
        return y
    
class InceptionV3_Light(nn.Module):
    def __init__(self, in_channels=1, num_classes=5):
        super().__init__()

        self.stem = InceptionStemV3(in_channels)
        
        #Reduced the number of InceptionA blocks from 3 to 2
        self.inception_a = nn.Sequential(
            InceptionA(in_channels=192, pool_features=32),
            InceptionA(in_channels=256, pool_features=64),
        )
        
        self.reduction_a = InceptionReductionA(in_channels=288) 
        
        #Reduced from 4 to 1 only block (removed the 3 intermediates)
        self.inception_b = nn.Sequential(
            InceptionB(in_channels=768, internal_filters=128),
        )
        
        self.reduction_b = InceptionReductionB(in_channels=768)
        
        # Reduced from 2 to 1 only block 
        self.inception_c = nn.Sequential(
            InceptionC(in_channels=1280),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.drop = nn.Dropout(p=0.5) 
        self.out = nn.Linear(2048, num_classes) 

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.drop(x)
        y = self.out(x)
        
        return y