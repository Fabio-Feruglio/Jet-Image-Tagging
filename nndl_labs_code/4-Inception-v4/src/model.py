import torch
from torch import nn
from model_blocks import StemBlock, Reduction_A, Reduction_B, A_block, B_block, C_block

class InceptionV4(nn.Module):

    def __init__(self):
        super().__init__()

        self.stem = StemBlock()
        self.inception_a = nn.Sequential(
            A_block(in_filters=384),
            A_block(in_filters=384),
            A_block(in_filters=384),
            A_block(in_filters=384),
        )
        self.reduction_a = Reduction_A(384)
        self.inception_b = nn.Sequential(
            B_block(in_filters=1024),
            B_block(in_filters=1024),
            B_block(in_filters=1024),
            B_block(in_filters=1024),
            B_block(in_filters=1024),
            B_block(in_filters=1024),
            B_block(in_filters=1024),
        )
        self.reduction_b = Reduction_B(1024)
        self.inception_c = nn.Sequential(
            C_block(in_filters=1536),
            C_block(in_filters=1536),
            C_block(in_filters=1536),
        )
        self.drop = nn.Dropout(p=0.2)
        self.out = nn.Linear(1536, 1)

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = x.reshape(x.shape[0], -1, 1536).mean(axis=1)
        x = self.drop(x)
        y = self.out(x)
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