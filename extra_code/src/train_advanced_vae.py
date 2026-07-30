import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
import wandb

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.variational_autoencoder_advanced import Encoder, Decoder

def VAE_loss_fn(reconstructed_x, x, mu, log_var, beta=1.0):
    # Sum over pixels instead of mean, helps with sparse data like jet images
    recon_loss = torch.nn.functional.mse_loss(reconstructed_x, x, reduction='sum') / x.shape[0]
    # KL Divergence
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    
    total_loss = recon_loss + beta * kl_div
    return total_loss, recon_loss.item(), kl_div.item()

def train_epoch(encoder, decoder, dataloader, optimizer, device, current_beta):
    encoder.train()
    decoder.train()
    losses, recon_losses, kl_losses = [], [], []

    train_iterator = tqdm(dataloader)
    for x_batch, _ in train_iterator:
        x_batch = x_batch.to(device)

        encoded, mu, log_var = encoder(x_batch)
        reconstructed_x = decoder(encoded)

        loss, recon, kl = VAE_loss_fn(reconstructed_x, x_batch, mu, log_var, beta=current_beta) 

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Loss: {loss.item():.2f} (Beta: {current_beta:.4f})")
        losses.append(loss.item())
        recon_losses.append(recon)
        kl_losses.append(kl)

    return np.mean(losses), np.mean(recon_losses), np.mean(kl_losses)

def val_epoch(encoder, decoder, dataloader, device, current_beta):
    encoder.eval()
    decoder.eval()
    losses = []
    with torch.no_grad():
        for x_batch, _ in dataloader:
            x_batch = x_batch.to(device)
            encoded, mu, log_var = encoder(x_batch)
            reconstructed_x = decoder(encoded) # Stocastico in validazione per la loss
            loss, _, _ = VAE_loss_fn(reconstructed_x, x_batch, mu, log_var, beta=current_beta)
            losses.append(loss.item())
            
    return np.mean(losses)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.save_dir, exist_ok=True)

    run = wandb.init(project="jet-tagging-anomaly-detection", name="train_vae_advanced", config=vars(args))
    
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        data_filepath=args.data_path, bg_classes=args.bg_classes,
        img_size=args.img_size, batch_size=args.batch_size, 
        num_workers=2, max_samples=args.max_samples
    )
    
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), 
                                 lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float('inf')
    
    for epoch in range(args.epochs):
        # KL Annealing: incrementa beta linearmente fino al valore max
        current_beta = args.target_beta * min(1.0, epoch / max(1, args.warmup_epochs))

        train_loss, recon, kl = train_epoch(encoder, decoder, train_dataloader, optimizer, device, current_beta)
        val_loss = val_epoch(encoder, decoder, valid_dataloader, device, current_beta)

        print(f'EPOCH {epoch+1} - Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}')
        wandb.log({"Epoch": epoch, "Loss/Train": train_loss, "Loss/Validation": val_loss, "Beta": current_beta, "KL": kl, "Recon": recon})

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {'encoder_state_dict': encoder.state_dict(), 'decoder_state_dict': decoder.state_dict()}
            torch.save(checkpoint, os.path.join(args.save_dir, 'vae_advanced_best.pth'))

    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--latent_space_dim', type=int, default=32) # Ridotto a 32 per creare un collo di bottiglia reale
    parser.add_argument('--target_beta', type=float, default=0.1) # Peso finale della KL
    parser.add_argument('--warmup_epochs', type=int, default=10) # Epoche per arrivare al target_beta
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    args = parser.parse_args()
    main(args)