import torch
from torch import nn
import torch.nn.functional as F
from .mininception import MinInception
from .resnet10 import ResNet10

class HybridEncoder(nn.Module):
    def __init__(self, latent_space_dim=128, proj_dim=64, in_features=1):
        super().__init__()
        self.resnet_branch = ResNet10(in_features=in_features, out_features=256)
        self.inception_branch = MinInception(in_channels=in_features, out_features=256)

        # Spazio latente z per la ricostruzione
        self.out = nn.Sequential(nn.Linear(512, latent_space_dim))
        
        # Projection Head p per il contrastive learning (mappa su un'ipersfera)
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
        
        z = self.out(combined_features)
        
        # Applichiamo la normalizzazione L2 all'output della projection head
        p = F.normalize(self.proj_head(z), dim=1)
        
        return z, p

class Decoder(nn.Module):
    # IDENTICO ALL'AUTOENCODER ORIGINALE[cite: 5]
    def __init__(self, latent_space_dim=128, im_size=128, base_channels=16):
        super().__init__()
        c = base_channels
        self.start_channels = c * 16

        self.decoder_lin = nn.Sequential(
            nn.Linear(in_features=latent_space_dim, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=self.start_channels * 4 * 4), 
            nn.ReLU(True),
        )
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.start_channels, 4, 4))
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(c * 16, c * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 8), nn.ReLU(True),
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4), nn.ReLU(True),
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2), nn.ReLU(True),
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c), nn.ReLU(True),
            nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1),
        )

    def forward(self, z):
        x = self.decoder_lin(z)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x