import torch
from torch import nn
from .mininception import MinInception
from .resnet10 import ResNet10

def reparameterize(mu, log_var):
    std = torch.exp(0.5 * log_var)
    eps = torch.randn_like(std)
    return mu + eps * std

class Encoder(nn.Module):
    def __init__(self, latent_space_dim, in_features=1):
        super().__init__()
        self.resnet_branch = ResNet10(in_features=in_features, out_features=256)
        self.inception_branch = MinInception(in_channels=in_features, out_features=256)

        self.mu = nn.Sequential(
            nn.Linear(512, latent_space_dim)
        )
        self.log_var = nn.Sequential(
            nn.Linear(512, latent_space_dim)
        )

    def forward(self, x):
        resnet_features = self.resnet_branch(x)
        inception_features = self.inception_branch(x)
        combined_features = torch.cat([resnet_features, inception_features], dim=1)
        mu = self.mu(combined_features)
        log_var = self.log_var(combined_features)
        z = reparameterize(mu, log_var)
        return z, mu, log_var

class Decoder(nn.Module):
    def __init__(self, latent_space_dim, im_size=128, base_channels=16):
        super().__init__()
        
        c = base_channels
        self.im_size = im_size
        self.start_channels = c * 16

        # Linear layer
        self.decoder_lin = nn.Sequential(
            nn.Linear(in_features=latent_space_dim, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=self.start_channels * 4 * 4), 
            nn.ReLU(True),
        )

        # Unflatten: [batch, channels, 4, 4]
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.start_channels, 4, 4))

        self.decoder_conv = nn.Sequential(
            # IN: if c=16, 256 channels, 4x4
            nn.ConvTranspose2d(c * 16, c * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(True),
            # OUT: 128 channels, 8x8
            
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(True),
            # OUT: 64 channels, 16x16
            
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(True),
            # OUT: 32 channels, 32x32
            
            nn.ConvTranspose2d(c * 2, c, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            # OUT: 16 channels, 64x64
            
            nn.ConvTranspose2d(c, 1, kernel_size=4, stride=2, padding=1),
            # OUT: 1 channel, 128x128
        )

    def forward(self, x):
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        return x