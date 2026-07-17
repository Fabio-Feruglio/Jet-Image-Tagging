import torch
from torch import nn
from ....classification.src.model.ensemble import EnsembleModel


class ContrastiveEncoder(nn.Module):
    def __init__(self, repr_space_dim, latent_space_dim):
        super().__init__()
        super().__init__()

        self.encoder = EnsembleModel()
        self.encoder.fc = nn.Sequential(
            nn.Linear(2048 + 1536, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, repr_space_dim), 
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(repr_space_dim, 256),
            nn.ReLU(),
            nn.Linear(256, latent_space_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return self.head(x)
    
