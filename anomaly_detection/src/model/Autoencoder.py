import torch
from torch import nn

JET_TYPES = ["g", "q", "t", "w", "z"]  # gluon, light quark, top, W boson, Z boson
LABEL_NAMES = {i: name for i, name in enumerate(JET_TYPES)}

class Encoder(nn.Module):
    def __init__(self, encoded_space_dim, base_channels=8):
        super().__init__()
        c = base_channels

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c),
            nn.ReLU(True),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 2),
            nn.ReLU(True),
            nn.Conv2d(c * 2, c * 4, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 4),
            nn.ReLU(True),
            nn.Conv2d(c * 4, c * 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 8),
            nn.ReLU(True),
            nn.Conv2d(c * 8, c * 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 16),
            nn.ReLU(True),
            nn.Conv2d(c * 16, c * 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(c * 16),
            nn.ReLU(True),
        )

        self.flatten = nn.Flatten(start_dim=1)

        self.encoder_lin = nn.Sequential(
            nn.Linear(in_features=c * 16 * 5 * 5, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=encoded_space_dim),
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        x = self.flatten(x)
        x = self.encoder_lin(x)
        return x


class Decoder(nn.Module):
    def __init__(self, encoded_space_dim, im_size=299, base_channels=8):
        super().__init__()
        if im_size != 299:
            raise ValueError("This Decoder's output_padding values were hand-computed for im_size=299")
        
        c = base_channels
        self.im_size = im_size
        self._start_channels = c * 16

        self.decoder_lin = nn.Sequential(
            nn.Linear(in_features=encoded_space_dim, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=self._start_channels * 5 * 5),
            nn.ReLU(True),
        )

        self.unflatten = nn.Unflatten(dim=1, unflattened_size=(self._start_channels, 5, 5))

        self.decoder_conv = nn.Sequential(
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
        x = self.decoder_lin(x)
        x = self.unflatten(x)
        x = self.decoder_conv(x)
        x = torch.sigmoid(x)
        return x