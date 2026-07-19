import torch
from torch import nn
from ....classification.src.model.ensemble import EnsembleModel
from ....classification.src.model.resnet import ResNet50

class ContrastiveEncoder_Light(nn.Module):
    def __init__(self, repr_space_dim, latent_space_dim):
        super().__init__()

        resnet_output = 2048
        self.encoder = ResNet50(num_classes=resnet_output)
        self.encoder.fc = nn.Sequential(
            nn.Linear(resnet_output, resnet_output),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(resnet_output, repr_space_dim), 
        )
        
        self.head = nn.Sequential(
            nn.ReLU(),
            nn.Linear(repr_space_dim, repr_space_dim),
            nn.ReLU(),
            nn.Linear(repr_space_dim, latent_space_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return self.head(x)


class ContrastiveEncoder(nn.Module):
    def __init__(self, repr_space_dim, latent_space_dim):
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
            nn.Linear(repr_space_dim, repr_space_dim),
            nn.ReLU(),
            nn.Linear(repr_space_dim, latent_space_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.fc(x)
        return self.head(x)
    
def contrastive_labeled_loss(z_i, z_j, temperature=0.5):
    batch_size = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    sim_matrix = torch.exp(torch.mm(z, z.t().contiguous()) / temperature)
    mask = (torch.ones_like(sim_matrix) - torch.eye(2 * batch_size, device=sim_matrix.device)).bool()
    sim_matrix = sim_matrix.masked_select(mask).view(2 * batch_size, -1)
    pos_sim = torch.exp(torch.sum(z_i * z_j, dim=-1) / temperature)
    pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
    loss = -torch.log(pos_sim / sim_matrix.sum(dim=-1))
    return loss.mean()