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


class BasicBlock(nn.Module):

    def __init__(self, in_filters, out_filters, stride=1, expansion=1):
        super().__init__()
        
        # Expansion factor for the output channels
        self.expansion = expansion

        # Main Path
        self.main_path = nn.Sequential(
            Conv2d_bn(in_filters = in_filters, out_filters = out_filters, kernel_size = 3, strides = stride, padding = "same", activation=True),
            Conv2d_bn(in_filters = out_filters, out_filters = out_filters * self.expansion, kernel_size = 3, strides = 1, padding = "same")
        )
        
        # Skip Connection
        self.shortcut = nn.Sequential()
        # If the input and output dimensions do not match, we need to adjust the shortcut path
        if stride != 1 or in_filters != out_filters * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_filters, out_filters * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_filters * self.expansion)
            )
        self.relu = nn.ReLU()
    
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

        out = self.main_path(x) + self.shortcut(x)
        out = self.relu(out)

        return out

class ResNet10(nn.Module):
    def __init__(self, in_features = 1,out_features = 512):
        super().__init__()
        self.apply(self._init_weights)

        self.network = nn.Sequential(
            
            # Stem
            nn.Conv2d(in_features, 64, kernel_size=7, stride=2, padding=3, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

            # Main blocks
            BasicBlock(64, 64, stride=1),
            nn.Dropout(0.2),
            BasicBlock(64, 128, stride=2),
            nn.Dropout(0.2),
            BasicBlock(128, 256, stride=2),
            nn.Dropout(0.2),

            #BasicBlock(256, 512, stride=2), # removed to reduce model size and complexity
            #nn.Dropout(0.2),


            nn.AdaptiveAvgPool2d((1, 1)), 
        )
        self.out = nn.Linear(256, out_features)
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
        y = self.network(x)
        y = torch.flatten(y, 1) 
        y = self.out(y)
        return y