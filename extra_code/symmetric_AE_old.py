import torch
from torch import nn
from .mininception import MinInception
from .resnet10 import ResNet10

class BasicBlockTranspose(nn.Module):
    """
    Specchio del BasicBlock di ResNet.
    Usa ConvTranspose2d per fare upsampling spaziale quando richiesto.
    """
    def __init__(self, in_filters, out_filters, stride=1):
        super().__init__()
        
        if stride == 2:
            self.main = nn.Sequential(
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(True),
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_filters)
            )
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_filters, out_filters, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_filters)
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_filters),
                nn.ReLU(True),
                nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(out_filters)
            )
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_filters, out_filters, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_filters)
            ) if in_filters != out_filters else nn.Identity()
            
        self.relu = nn.ReLU(True)

    def forward(self, x):
        return self.relu(self.main(x) + self.shortcut(x))


class InceptionTransposeBlock(nn.Module):
    """
    Specchio logico del blocco Inception. Usa diramazioni parallele 
    per elaborare diverse feature e le ri-concatena garantendo le dimensioni.
    """
    def __init__(self, in_channels, out_channels, upsample=False):
        super().__init__()
        branch_c = out_channels // 4
        rem = out_channels - (branch_c * 3)

        if upsample:
            layer = lambda c_in, c_out: nn.ConvTranspose2d(c_in, c_out, kernel_size=4, stride=2, padding=1, bias=False)
        else:
            layer = lambda c_in, c_out: nn.Conv2d(c_in, c_out, kernel_size=3, stride=1, padding=1, bias=False)

        self.b1 = nn.Sequential(layer(in_channels, branch_c), nn.BatchNorm2d(branch_c), nn.ReLU(True))
        self.b2 = nn.Sequential(layer(in_channels, branch_c), nn.BatchNorm2d(branch_c), nn.ReLU(True))
        self.b3 = nn.Sequential(layer(in_channels, branch_c), nn.BatchNorm2d(branch_c), nn.ReLU(True))
        self.b4 = nn.Sequential(layer(in_channels, rem), nn.BatchNorm2d(rem), nn.ReLU(True))

    def forward(self, x):
        return torch.cat([self.b1(x), self.b2(x), self.b3(x), self.b4(x)], dim=1)


# ==========================================
# BRANCH DECODERS
# ==========================================
class MiniResNetDecoder(nn.Module):
    def __init__(self, in_features=256):
        super().__init__()
        # Reverse dell'AdaptiveAvgPool (1x1) a un tensor 8x8
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256 * 8 * 8),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(1, (256, 8, 8))
        
        # 8x8 -> 16x16
        self.up1 = BasicBlockTranspose(256, 128, stride=2)
        # 16x16 -> 32x32
        self.up2 = BasicBlockTranspose(128, 64, stride=2)
        # 32x32 -> 32x32
        self.up3 = BasicBlockTranspose(64, 64, stride=1)
        
        # Reverse Stem: 32x32 -> 64x64 -> 128x128
        self.stem_up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.stem_up(x)
        return x


class MinInceptionDecoder(nn.Module):
    def __init__(self, in_features=256):
        super().__init__()
        # Reverse dell'AdaptiveAvgPool (1x1) a un tensor 16x16
        self.fc = nn.Sequential(
            nn.Linear(in_features, 256 * 16 * 16),
            nn.ReLU(True)
        )
        self.unflatten = nn.Unflatten(1, (256, 16, 16))
        
        # 16x16 -> 32x32
        self.up1 = InceptionTransposeBlock(256, 128, upsample=True)
        # 32x32 -> 32x32
        self.up2 = InceptionTransposeBlock(128, 64, upsample=False)
        # 32x32 -> 32x32
        self.up3 = InceptionTransposeBlock(64, 32, upsample=False)
        
        # Reverse Stem: 32x32 -> 64x64 -> 128x128
        self.stem_up = nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False), 
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def forward(self, x):
        x = self.fc(x)
        x = self.unflatten(x)
        x = self.up1(x)
        x = self.up2(x)
        x = self.up3(x)
        x = self.stem_up(x)
        return x


# ==========================================
# FULL ENCODER & DECODER
# ==========================================
class Encoder(nn.Module):
    def __init__(self, latent_space_dim, in_features=1):
        super().__init__()
        self.resnet_branch = ResNet10(in_features=in_features, out_features=256)
        self.inception_branch = MinInception(in_channels=in_features, out_features=256)

        self.out = nn.Sequential(
            nn.Linear(512, latent_space_dim)
        )

    def forward(self, x):
        resnet_features = self.resnet_branch(x)
        inception_features = self.inception_branch(x)
        combined_features = torch.cat([resnet_features, inception_features], dim=1)
        x = self.out(combined_features)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_space_dim):
        super().__init__()
        # Espansione a 512 per splittare tra i due branch
        self.expand = nn.Sequential(
            nn.Linear(latent_space_dim, 512),
            nn.ReLU(True)
        )
        
        self.resnet_dec = MiniResNetDecoder(in_features=256)
        self.inception_dec = MinInceptionDecoder(in_features=256)
        
        # Merge layer finale (32+32 = 64 canali -> 1 canale immagine)
        self.merge = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, z):
        x = self.expand(z)
        x_res, x_inc = torch.split(x, 256, dim=1)
        
        out_res = self.resnet_dec(x_res)
        out_inc = self.inception_dec(x_inc)
        
        out_combined = torch.cat([out_res, out_inc], dim=1)
        final_image = self.merge(out_combined)
        
        return final_image