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



### Inception Stem (modified for 1 channel input): (299, 299, 1) -> (35, 35, 384)
class InceptionStemBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.first_block = nn.Sequential(
           Conv2d_bn(in_filters=1, out_filters=32, kernel_size=(3,3), strides=2, padding="valid"),
           Conv2d_bn(in_filters=32, out_filters=32, kernel_size=(3,3), strides=1, padding="valid"),
           Conv2d_bn(in_filters=32, out_filters=64, kernel_size=(3,3), strides=1, padding="same"),
        )
        self.first_left = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)
        self.first_right = Conv2d_bn(in_filters=64, out_filters=96, kernel_size=(3,3), strides=2, padding="valid")
        self.second_left =  nn.Sequential(
            Conv2d_bn(in_filters=160, out_filters=64, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=(3,3), strides=1, padding="valid"),
        )
        self.second_right =  nn.Sequential(
            Conv2d_bn(in_filters=160, out_filters=64, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=64, kernel_size=(7,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=64, kernel_size=(1,7), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=(3,3), strides=1, padding="valid"),
        )
        self.third_left = Conv2d_bn(in_filters=192, out_filters=192, kernel_size=(3,3), strides=2, padding="valid")
        self.third_right = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)

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
        x = self.first_block(x)

        xl1 = self.first_left(x)
        xr1 = self.first_right(x)
        x = torch.cat([xl1, xr1], dim=1)

        xl2 = self.second_left(x)
        xr2 = self.second_right(x)
        x = torch.cat([xl2,xr2], dim=1)

        x = torch.cat([self.third_left(x), self.third_right(x)], dim=1)
        return x
    
### Inception A Block: (35, 35, 384) -> (35, 35, 384)
class InceptionA_block(nn.Module):

    def __init__(self, in_filters):
        super().__init__()

        self.avg_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=96, kernel_size=(1,1), strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=96, kernel_size=(1,1), strides=1, padding="same")
        self.three_by_three_block =  nn.Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=64, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=(3,3), strides=1, padding="same"),
        )
        self.five_by_five =  nn.Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=64, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=64, out_filters=96, kernel_size=(3,3), strides=1, padding="same"),
            Conv2d_bn(in_filters=96, out_filters=96, kernel_size=(3,3), strides=1, padding="same"),
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
        x1 = self.avg_block(x)
        x2 = self.one_by_one_block(x)
        x3 = self.three_by_three_block(x)
        x4 = self.five_by_five(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)

        return x
    
### Inception B Block: (17, 17, 1024) -> (17, 17, 1024)
class InceptionB_block(nn.Module):

    def __init__(self, in_filters):
        super().__init__()
        self.avg_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=128, kernel_size=(1,1), strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=(1,1), strides=1, padding="same")

        self.seven_by_seven_block =  nn.Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=(1,7), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=(7,1), strides=1, padding="same"),
        )

        self.thirteen_by_thirteen_block =  nn.Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=192, kernel_size=(1,7), strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=(7,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=224, kernel_size=(1,7), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=(7,1), strides=1, padding="same"),
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
        x1 = self.avg_block(x)
        x2 = self.one_by_one_block(x)
        x3 = self.seven_by_seven_block(x)
        x4 = self.thirteen_by_thirteen_block(x)
        x = torch.cat([x1, x2, x3, x4], dim=1)
        return x
    
### Inception C Block: (8, 8, 1536) -> (8, 8, 1536)
class InceptionC_block(nn.Module):

    def __init__(self, in_filters):
        super().__init__()

        self.avg_block = nn.Sequential(
            nn.AvgPool2d(kernel_size=(3,3), stride=1, padding=1),
            Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=(1,1), strides=1, padding="same"),
        )
        self.one_by_one_block = Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=(1, 1), strides=1, padding="same")

        self.branch_a =  Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=(1, 1), strides=1, padding="same")
        self.branch_a_left = Conv2d_bn(in_filters=384, out_filters=256, kernel_size=(1, 3), strides=1, padding="same")
        self.branch_a_right = Conv2d_bn(in_filters=384, out_filters=256, kernel_size=(3, 1), strides=1, padding="same")

        self.branch_b =  nn.Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=(1, 1), strides=1, padding="same"),
            Conv2d_bn(in_filters=384, out_filters=448, kernel_size=(1, 3), strides=1, padding="same"),
            Conv2d_bn(in_filters=448, out_filters=512, kernel_size=(3, 1), strides=1, padding="same"),
        )


        self.branch_b_left = Conv2d_bn(in_filters=512, out_filters=256, kernel_size=(1, 3), strides=1, padding="same")
        self.branch_b_right = Conv2d_bn(in_filters=512, out_filters=256, kernel_size=(3, 1), strides=1, padding="same")
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
        x1 = self.avg_block(x)
        x2 = self.one_by_one_block(x)
        x3 = self.branch_a(x)
        x3a = self.branch_a_left(x3)
        x3b = self.branch_a_right(x3)
        x4 = self.branch_b(x)
        x4a = self.branch_b_left(x4)
        x4b = self.branch_b_right(x4)
        x = torch.cat([x1, x2, x3a, x3b, x4a, x4b], dim=1)
        return x

### Inception Reduction A Block: (35, 35, 384) -> (17, 17, 1024)
class InceptionReductionA(nn.Module):

    def __init__(self, in_filters):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)
        self.central_block = Conv2d_bn(in_filters=in_filters, out_filters=384, kernel_size=(3,3), strides=2, padding="valid")
        self.right_block =  nn.Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=224, kernel_size=(3,3), strides=1, padding="same"),
            Conv2d_bn(in_filters=224, out_filters=256, kernel_size=(3,3), strides=2, padding="valid"),
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
        x1 = self.max_pool(x)
        x2 = self.central_block(x)
        x3 = self.right_block(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x
    
### Inception Reduction B Block: (17, 17, 1024) -> (8, 8, 1536)
class InceptionReductionB(nn.Module):

    def __init__(self, in_filters):
        super().__init__()

        self.max_pool = nn.MaxPool2d(kernel_size=(3,3), stride=2, padding=0)
        self.central_block = nn.Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=192, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=192, out_filters=192, kernel_size=(3,3), strides=2, padding="valid"),
        )
        self.right_block =  nn.Sequential(
            Conv2d_bn(in_filters=in_filters, out_filters=256, kernel_size=(1,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=256, out_filters=256, kernel_size=(1,7), strides=1, padding="same"),
            Conv2d_bn(in_filters=256, out_filters=320, kernel_size=(7,1), strides=1, padding="same"),
            Conv2d_bn(in_filters=320, out_filters=320, kernel_size=(3,3), strides=2, padding="valid"),

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
        x1 = self.max_pool(x)
        x2 = self.central_block(x)
        x3 = self.right_block(x)
        x = torch.cat([x1, x2, x3], dim=1)
        return x