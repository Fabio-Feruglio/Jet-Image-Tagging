import torch
from torch import nn
from .ensemble import EnsembleModel
from .resnet import ResNet50

class Vencoder_Light(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        resnet_output = 2048
        self.Vencoder = ResNet50(num_classes=resnet_output)
        self.Vencoder.out = nn.Sequential(
            nn.Linear(resnet_output, resnet_output),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 512),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(in_features=512, out_features=encoded_space_dim)
        self.fc_var = nn.Linear(in_features=512, out_features=encoded_space_dim)

    def forward(self, x):
        x = self.Vencoder(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var

class VEnconder_Ensemble(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        self.Vencoder = EnsembleModel()
        self.Vencoder.fc = nn.Sequential(
            nn.Linear(2048 + 2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(in_features=1024, out_features=encoded_space_dim)
        self.fc_log_var = nn.Linear(in_features=1024, out_features=encoded_space_dim)

    def forward(self, x):
        x = self.Vencoder(x)
        mu = self.fc_mu(x)
        log_var = self.fc_log_var(x)
        return mu, log_var




class VDecoder_Ensemble(nn.Module):
    def __init__(self, encoded_space_dim, im_size=299, base_channels=8):
        super().__init__()
        if im_size != 299:
            raise ValueError("This Decoder's output_padding values were hand-computed for im_size=299")
        
        c = base_channels
        self.im_size = im_size
        self.start_channels = c * 16

        self.Vdecoder_lin = nn.Sequential(
            nn.Linear(in_features=encoded_space_dim, out_features=512),
            nn.ReLU(True),
            nn.Linear(in_features=512, out_features=self.start_channels * 5 * 5),
            nn.ReLU(True)
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self.start_channels, 5, 5))

        self.Vdecoder_deconv = nn.Sequential(
            nn.ConvTranspose2d(c * 16, c * 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c * 16),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 16, c * 8, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 8, c * 4, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 4, c * 2, kernel_size=3, stride=2, padding=1, output_padding=0),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(c * 2, c, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            nn.ConvTranspose2d(c, 1, kernel_size=3, stride=2, padding=1, output_padding=0),
        )

    def forward(self, x):
            x = self.Vdecoder_lin(x)
            x = self.unflatten(x)
            x = self.Vdecoder_deconv(x)
            return x
        
class Vencoder_Ensemble_Light(nn.Module):
    def __init__(self, encoded_space_dim):
        super().__init__()

        self.Vencoder = EnsembleModel()
        self.Vencoder.fc = nn.Sequential(
            nn.Linear(2048 + 2048, 2048),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(2048, 1024),
            nn.ReLU()
        )
        self.fc_mu = nn.Linear(in_features=1024, out_features=encoded_space_dim)
        self.fc_var = nn.Linear(in_features=1024, out_features=encoded_space_dim)

    def forward(self, x):
        x = self.Vencoder(x)
        mu = self.fc_mu(x)
        var = self.fc_var(x)
        return mu, var

class VAE_Ensemble(nn.Module):
    def __init__(self, encoded_space_dim, im_size=299, base_channels=8):
        super().__init__()
        self.encoder = VEnconder_Ensemble(encoded_space_dim=encoded_space_dim)
        self.decoder = VDecoder_Ensemble(encoded_space_dim=encoded_space_dim, im_size=im_size, base_channels=base_channels)

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var
    
class VAE_Ensemble_Light(nn.Module):
    def __init__(self, encoded_space_dim, im_size=299, base_channels=8):
        super().__init__()
        self.encoder = Vencoder_Ensemble_Light(encoded_space_dim=encoded_space_dim)
        self.decoder = VDecoder_Ensemble(encoded_space_dim=encoded_space_dim, im_size=im_size, base_channels=base_channels)

    def reparameterize(self, mu, var):
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu, var = self.encoder(x)
        z = self.reparameterize(mu, var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, var

def test_func():
    print("test")