import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset.dataloader import get_dataloaders
# Assicurati di importare anche TransformHead
from model.other_models_attempt.miniVAE import Encoder, Decoder, TransformHead

### CUSTOM LOSS FUNC FOR VAE ###
def VAE_loss_fn(reconstructed_x, x, mu, log_var, sigma=1.0):
    # 1. MSE puro (Media su tutto il batch e tutti i pixel)
    recon_loss = torch.nn.functional.mse_loss(reconstructed_x, x, reduction='mean') / (sigma**2)

    # 2. KL Divergence (Media sul batch)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    # 3. Dividiamo la KL per il numero di pixel per bilanciarla con la media della MSE
    num_pixels = x.shape[1] * x.shape[2] * x.shape[3]
    kl_div_scaled = kl_div / num_pixels

    return recon_loss + kl_div_scaled

### SUPERVISED CONTRASTIVE LOSS ###
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        """
        features: Vettori proiettati e normalizzati L2, shape [batch_size, proj_dim]
        labels: Etichette delle classi, shape [batch_size]
        """
        device = features.device
        batch_size = features.shape[0]

        # 1. Calcolo della matrice di similarità (prodotto scalare tra vettori normalizzati = similarità coseno)
        sim_matrix = torch.matmul(features, features.T) / self.temperature

        # 2. Creazione della maschera per identificare i campioni della stessa classe
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        # 3. Rimuoviamo la diagonale (non vogliamo contrastare un'immagine con se stessa)
        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask

        # 4. Stabilizzazione numerica
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        # 5. Calcolo delle probabilità
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        # 6. Media della log-likelihood solo sulle coppie positive
        mask_sum = mask.sum(1)
        # Evitiamo divisioni per zero se una classe ha un solo campione nel batch
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        # 7. La loss finale
        loss = -mean_log_prob_pos.mean()
        return loss

### TRAINING ###
def train_epoch(encoder, decoder, transform_head, dataloader, vae_loss_fn, supcon_loss_fn, optimizer, device, contrastive_weight):
    encoder.train()
    decoder.train()
    transform_head.train()
    losses = []

    train_iterator = tqdm(dataloader)
    for x_batch, label_batch in train_iterator:
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)

        # Normalizzazione dei dati (assicurati di mantenerla come discusso precedentemente)
        if x_batch.max() > 1.0:
            x_batch = x_batch / 255.0

        # Forward pass
        encoded, mu, log_var = encoder(x_batch)
        reconstructed_x = decoder(encoded)
        
        # Estrazione delle features proiettate usando il vettore deterministico 'mu'
        proj_features = transform_head(mu)

        # Loss computation
        vae_loss = vae_loss_fn(reconstructed_x, x_batch, mu, log_var) 
        c_loss = supcon_loss_fn(proj_features, label_batch)
        
        # Loss combinata
        loss = vae_loss + (contrastive_weight * c_loss)

        # Backward pass
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Train loss: {loss.item():.4f} (VAE: {vae_loss.item():.4f}, SupCon: {c_loss.item():.4f})")
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    return avg_loss

### VALIDATION ###
def val_epoch(encoder, decoder, transform_head, dataloader, vae_loss_fn, supcon_loss_fn, device, contrastive_weight):
    encoder.eval()
    decoder.eval()
    transform_head.eval()
    losses = []

    with torch.no_grad():
        val_iterator = tqdm(dataloader)

        for x_batch, label_batch in val_iterator:
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            if x_batch.max() > 1.0:
                x_batch = x_batch / 255.0

            encoded, mu, log_var = encoder(x_batch)
            reconstructed_x = decoder(encoded)
            proj_features = transform_head(mu)

            vae_loss = vae_loss_fn(reconstructed_x, x_batch, mu, log_var)
            c_loss = supcon_loss_fn(proj_features, label_batch)
            
            loss = vae_loss + (contrastive_weight * c_loss)

            losses.append(loss.item())
            val_iterator.set_description(f"Val loss: {loss.item():.4f}")
            
    avg_loss = np.mean(losses)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    # 1. Setup Device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Selected Device: {device}')
    
    # 2. Folders creation
    os.makedirs(args.save_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard_logs_anomaly_detection'))

    wandb_run_id = None
    if args.resume_from and os.path.isfile(args.resume_from):
        temp_checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        if 'wandb_run_id' in temp_checkpoint:
            wandb_run_id = temp_checkpoint['wandb_run_id']
            print(f"ID: {wandb_run_id}")

    run = wandb.init(
        project = "jet-tagging-anomaly-detection-vae-attempt",
        name = f"train_vae_supcon_lr{args.lr}",
        config = vars(args),
        id = wandb_run_id,     
        resume = "allow"                                     
    )
    
    # 3. Load dataloaders 
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        data_filepath = args.data_path, 
        bg_classes = args.bg_classes,
        img_size = args.img_size, 
        batch_size = args.batch_size, 
        num_workers = min(4, os.cpu_count() or 1),
        max_samples = args.max_samples
    )
    
    # 4. Initialize model, heads and loss functions
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    transform_head = TransformHead(latent_space_dim=args.latent_space_dim, proj_dim=args.proj_dim).to(device)
    
    vae_loss_fn = VAE_loss_fn
    supcon_loss_fn = SupervisedContrastiveLoss(temperature=args.temperature)

    # 5. Define an optimizer (INCLUSO IL TRANSFORM HEAD)
    lr = args.lr 
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': lr},
        {'params': decoder.parameters(), 'lr': lr},
        {'params': transform_head.parameters(), 'lr': lr}
    ], weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float('inf')
    patience = args.patience 
    no_improvement_epochs = 0

    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint from '{args.resume_from}' ...")
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
            
            encoder.load_state_dict(checkpoint['encoder_state_dict'])
            decoder.load_state_dict(checkpoint['decoder_state_dict'])
            transform_head.load_state_dict(checkpoint['transform_head_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            if 'no_improvement_epochs' in checkpoint:
                no_improvement_epochs = checkpoint['no_improvement_epochs']
            print(f"Training resumed from {start_epoch}")
        else:
            print(f"No file found in '{args.resume_from}', starting from epoch = 0.")
    
    # 6. Training cycle
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(encoder, decoder, transform_head, train_dataloader, vae_loss_fn, supcon_loss_fn, optimizer, device, args.contrastive_weight)
        val_loss = val_epoch(encoder, decoder, transform_head, valid_dataloader, vae_loss_fn, supcon_loss_fn, device, args.contrastive_weight)

        print(f'EPOCH {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}')
        print(f'EPOCH {epoch+1}/{args.epochs} - Validation Loss: {val_loss:.4f}')

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.flush()

        wandb.log({
            "Epoch": epoch,
            "Loss/Train": train_loss,
            "Loss/Validation": val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            no_improvement_epochs = 0
            is_best = True
        else:
            no_improvement_epochs += 1
            is_best = False

        checkpoint_dict = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'transform_head_state_dict': transform_head.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improvement_epochs': no_improvement_epochs,
            'wandb_run_id': run.id
        }
        torch.save(checkpoint_dict, os.path.join(args.save_dir, 'miniVAE_latest.pth'))
        if is_best:
            torch.save(checkpoint_dict, os.path.join(args.save_dir, 'miniVAE_best.pth'))

        if no_improvement_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    writer.close()
    wandb.finish()
    print(f'Training completed. Best model saved in {os.path.join(args.save_dir, "miniVAE_best.pth")}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the ensemble model for jet image classification")
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1], help='Classes to consider as background (e.g. 0 1)')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch dimension')
    parser.add_argument('--img_size', type=int, default=299, help='Image size for resizing')
    parser.add_argument('--latent_space_dim', type=int, default=128, help='Dimension of the latent space')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay (L2 regularization) factor')
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to use for training")
    parser.add_argument('--data_path', type=str, default='./dataset.h5', help='Path to the dataset file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory for model/results saving')
    parser.add_argument('--resume_from', type=str, default=None, help="Path to weights already trained to resume training")
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for improvement before stopping')
    
    # NUOVI ARGOMENTI PER LA LOSS CONTRASTIVA
    parser.add_argument('--proj_dim', type=int, default=64, help='Dimensione dell output della TransformHead')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperatura per la SupCon Loss')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='Peso da dare alla loss contrastiva rispetto a quella del VAE')

    args = parser.parse_args()
    main(args)