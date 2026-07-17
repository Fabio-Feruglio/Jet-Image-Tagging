import os
import torch
from torch import nn

from .inceptionv3 import InceptionV3
from .resnet import ResNet50

class EnsembleModel(nn.Module):
    def __init__(self, num_classes = 5, resnet_path=None, inception_path=None, device='cpu'):
        super().__init__()
        self.resnet = ResNet50(num_classes = num_classes)
        self.inception = InceptionV3(num_classes = num_classes)

        

        if resnet_path and os.path.exists(resnet_path):
            print(f"Load ResNEt weights from {resnet_path}")
            checkpoint = torch.load(resnet_path, map_location=device, weights_only=False)
            self.resnet.load_state_dict(checkpoint['model_state_dict'])
            
        if inception_path and os.path.exists(inception_path):
            print(f"Load Inception weights from {inception_path}")
            checkpoint = torch.load(inception_path, map_location=device, weights_only=False)
            self.inception.load_state_dict(checkpoint['model_state_dict'])

        self.resnet.out = nn.Identity()
        self.inception.out = nn.Identity()    


        self.fc = nn.Sequential(
            nn.Linear(2048 + 2048, 512),
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
    