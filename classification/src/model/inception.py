import torch
from torch import nn
from model_blocks import InceptionReductionA, InceptionReductionB, InceptionA_block, InceptionB_block, InceptionC_block, InceptionStemBlock


class InceptionV4(nn.Module):

    def __init__(self, num_classes = 5):
        super().__init__()

        self.stem = InceptionStemBlock()
        self.inception_a = nn.Sequential(
            InceptionA_block(in_filters=384),
            InceptionA_block(in_filters=384),
            InceptionA_block(in_filters=384),
            InceptionA_block(in_filters=384),
        )
        self.reduction_a = InceptionReductionA(384)
        self.inception_b = nn.Sequential(
            InceptionB_block(in_filters=1024),
            InceptionB_block(in_filters=1024),
            InceptionB_block(in_filters=1024),
            InceptionB_block(in_filters=1024),
            InceptionB_block(in_filters=1024),
            InceptionB_block(in_filters=1024),
            InceptionB_block(in_filters=1024),
        )
        self.reduction_b = InceptionReductionB(1024)
        self.inception_c = nn.Sequential(
            InceptionC_block(in_filters=1536),
            InceptionC_block(in_filters=1536),
            InceptionC_block(in_filters=1536),
        )
        self.drop = nn.Dropout(p = 0.8) # mitigates overfitting
        self.out: nn.Module = nn.Linear(1536, num_classes) # adapted to get 5 classes output
        self.softmax: nn.Module = nn.Softmax(dim = 1) # normalizes the output to prob distribution

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.stem(x)
        x = self.inception_a(x)
        x = self.reduction_a(x)
        x = self.inception_b(x)
        x = self.reduction_b(x)
        x = self.inception_c(x)
        x = x.mean(dim = [2, 3]) 
        x = self.drop(x)
        y = self.out(x)
        y = self.softmax(y)
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