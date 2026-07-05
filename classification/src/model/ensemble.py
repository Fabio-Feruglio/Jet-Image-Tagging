import torch
from torch import nn

from inception import InceptionV4
from resnet import ResNet50

class EnsembleModel(nn.Module):
    def __init__(self, num_classes = 5):
        super().__init__()
        self.resnet = ResNet50(num_classes = num_classes)
        self.inception = InceptionV4(num_classes = num_classes)

        self.resnet.out = nn.Identity()
        self.inception.out = nn.Identity()


        self.fc = nn.Sequential(
            nn.Linear(2048 + 1536, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        resnet_out = self.resnet(x)
        inception_out = self.inception(x)
        combined_out = torch.cat((resnet_out, inception_out), dim=1)
        final_out = self.fc(combined_out)

        return final_out
    
    def _init_weights(self, module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()
                
        if isinstance(module, torch.nn.Conv2d):
            torch.nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                module.bias.data.zero_()