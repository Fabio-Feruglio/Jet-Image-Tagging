import torch
from torch import nn
import torch.nn.functional as F
from .mininception import MinInception
from .resnet10 import ResNet10

# Evitiamo di duplicare codice: importiamo il Decoder direttamente dal tuo Autoencoder standard
from .autoencoder import Decoder

class HybridEncoder(nn.Module):
    def __init__(self, latent_space_dim=128, proj_dim=64, in_features=1):
        super().__init__()
        self.resnet_branch = ResNet10(in_features=in_features, out_features=256)
        self.inception_branch = MinInception(in_channels=in_features, out_features=256)

        # Spazio latente z per la ricostruzione dell'immagine (MSE)
        self.out = nn.Sequential(
            nn.Linear(512, latent_space_dim)
        )
        
        # Projection Head p per il Contrastive Learning (mappa su un'ipersfera per calcolare distanze angolari)
        self.proj_head = nn.Sequential(
            nn.Linear(latent_space_dim, latent_space_dim),
            nn.BatchNorm1d(latent_space_dim),
            nn.ReLU(inplace=True),
            nn.Linear(latent_space_dim, proj_dim)
        )

    def forward(self, x):
        resnet_features = self.resnet_branch(x)
        inception_features = self.inception_branch(x)
        combined_features = torch.cat([resnet_features, inception_features], dim=1)
        
        # 1. Output per il Decoder
        z = self.out(combined_features)
        
        # 2. Output per la loss SupCon (normalizzazione L2 obbligatoria per mappare sulla superficie della sfera)
        p = F.normalize(self.proj_head(z), dim=1)
        
        return z, p