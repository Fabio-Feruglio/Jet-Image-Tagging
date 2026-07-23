import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import wandb

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.autoencoder import Encoder, Decoder

# --- 1. FUNZIONI SPECIFICHE PER DAE ---

def apply_spatial_dropout(x, drop_prob=0.2):
    """
    Simula il malfunzionamento dei sensori azzerando casualmente 
    una percentuale dei pixel dell'immagine.
    """
    mask = (torch.rand_like(x) > drop_prob).float()
    return x * mask

def weighted_mse_loss(reconstructed_x, original_x, active_weight=5.0):
    """
    Calcola l'MSE dando un peso maggiore ai pixel che nell'originale hanno energia > 0.
    """
    # MSE pixel per pixel (senza riduzione)
    mse_per_pixel = F.mse_loss(reconstructed_x, original_x, reduction='none')
    
    # Crea una maschera: 1 se il pixel originale ha energia, 0 altrimenti
    active_mask = (original_x > 0).float()
    
    # I pixel vuoti avranno peso 1.0, quelli attivi avranno peso active_weight
    weights = 1.0 + (active_weight - 1.0) * active_mask
    
    # Applica i pesi e calcola la media
    weighted_mse = mse_per_pixel * weights
    return weighted_mse.mean()

# --- 2. TRAINING LOOP ---

def train_epoch(encoder, decoder, dataloader, optimizer, device, active_weight, noise_factor):
    encoder.train()
    decoder.train()
    losses = []

    train_iterator = tqdm(dataloader, desc="Training")
    for x_batch, _ in train_iterator:
        x_batch = x_batch.to(device)
        
        # Corrompiamo l'input
        x_corrupted = apply_spatial_dropout(x_batch, drop_prob=noise_factor)

        # Il modello cerca di ricostruire a partire dall'immagine corrotta
        encoded = encoder(x_corrupted)
        reconstructed_x = decoder(encoded)

        # La loss si calcola rispetto all'immagine ORIGINALE PULITA
        loss = weighted_mse_loss(reconstructed_x, x_batch, active_weight=active_weight) 

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_postfix({"Loss": f"{loss.item():.4f}"})
        losses.append(loss.item())

    return np.mean(losses)

def val_epoch(encoder, decoder, dataloader, device, active_weight):
    encoder.eval()
    decoder.eval()
    losses = []

    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            # In validazione non applichiamo rumore
            encoded = encoder(x_batch)
            reconstructed_x = decoder(encoded)
            loss = weighted_mse_loss(reconstructed_x, x_batch, active_weight=active_weight)
            losses.append(loss.item())
            
    return np.mean(losses)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    run = wandb.init(project="jet-tagging-anomaly-detection", name="train_dae_weighted", config=vars(args))
    
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        data_filepath=args.data_path, bg_classes=args.bg_classes,
        img_size=args.img_size, batch_size=args.batch_size, max_samples=args.max_samples
    )
    
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                                 lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(encoder, decoder, train_dataloader, optimizer, device, args.active_weight, args.noise_factor)
        val_loss = val_epoch(encoder, decoder, valid_dataloader, device, args.active_weight)

        print(f'EPOCH {epoch+1} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')
        wandb.log({"Epoch": epoch, "Loss/Train": train_loss, "Loss/Validation": val_loss})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}, 
                       os.path.join(args.save_dir, 'dae_best.pth'))

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--latent_space_dim', type=int, default=128)
    parser.add_argument('--active_weight', type=float, default=5.0, help="Moltiplicatore MSE per i pixel attivi")
    parser.add_argument('--noise_factor', type=float, default=0.2, help="Percentuale di pixel da azzerare")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    main(args)