import argparse
import os
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.miniVAE import Encoder, Decoder, TransformHead

### LOSS FUNC AVANZATA: Weighted MSE + Beta-VAE ###
def VAE_loss_fn(reconstructed_x, x, mu, log_var, beta=0.01):
    # Weighted MSE: Penalizziamo di più gli errori sui pixel attivi del jet
    # Se il pixel originale è > 0 (sfondo nero ignorato), moltiplichiamo l'errore per 10
    weight = torch.where(x > 0.0, 10.0, 1.0)
    
    # Errore quadratico pesato
    squared_error = weight * (reconstructed_x - x)**2
    recon_loss = squared_error.mean()

    # KL Divergence scalata sui pixel
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    num_pixels = x.shape[1] * x.shape[2] * x.shape[3]
    kl_div_scaled = kl_div / num_pixels

    # Combinazione tramite parametro Beta
    final_loss = recon_loss + (beta * kl_div_scaled)
    
    return final_loss, recon_loss, kl_div_scaled

### SUPERVISED CONTRASTIVE LOSS ###
class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]

        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        logits_mask = torch.scatter(
            torch.ones_like(mask), 
            1, 
            torch.arange(batch_size).view(-1, 1).to(device), 
            0
        )
        mask = mask * logits_mask

        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-8)

        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum

        loss = -mean_log_prob_pos.mean()
        return loss

### TRAINING ###
def train_epoch(encoder, decoder, transform_head, dataloader, supcon_loss_fn, optimizer, device, args):
    encoder.train()
    decoder.train()
    transform_head.train()
    losses = []

    train_iterator = tqdm(dataloader)
    for x_batch, label_batch in train_iterator:
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)

        if x_batch.max() > 1.0:
            x_batch = x_batch / 255.0

        encoded, mu, log_var = encoder(x_batch)
        reconstructed_x = decoder(encoded)
        proj_features = transform_head(mu)

        # Loss computation
        vae_loss, r_loss, kl = VAE_loss_fn(reconstructed_x, x_batch, mu, log_var, beta=args.beta) 
        c_loss = supcon_loss_fn(proj_features, label_batch)
        
        loss = vae_loss + (args.contrastive_weight * c_loss)

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Loss: {loss.item():.4f} (R:{r_loss.item():.4f}|KL:{kl.item():.4f}|SupCon:{c_loss.item():.4f})")
        losses.append(loss.item())

    return np.mean(losses)

### VALIDATION ###
def val_epoch(encoder, decoder, transform_head, dataloader, supcon_loss_fn, device, args):
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

            vae_loss, _, _ = VAE_loss_fn(reconstructed_x, x_batch, mu, log_var, beta=args.beta)
            c_loss = supcon_loss_fn(proj_features, label_batch)
            
            loss = vae_loss + (args.contrastive_weight * c_loss)
            losses.append(loss.item())
            
    avg_loss = np.mean(losses)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Selected Device: {device}')
    
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard_logs_anomaly_detection'))

    wandb_run_id = None
    if args.resume_from and os.path.isfile(args.resume_from):
        temp_checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        if 'wandb_run_id' in temp_checkpoint:
            wandb_run_id = temp_checkpoint['wandb_run_id']

    run = wandb.init(
        project = "jet-tagging-anomaly-detection-vae-attempt",
        name = f"train_vae_advanced_lr{args.lr}_beta{args.beta}",
        config = vars(args),
        id = wandb_run_id,     
        resume = "allow"                                     
    )
    
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        data_filepath = args.data_path, 
        bg_classes = args.bg_classes,
        img_size = args.img_size, 
        batch_size = args.batch_size, 
        num_workers = min(4, os.cpu_count() or 1),
        max_samples = args.max_samples
    )
    
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    transform_head = TransformHead(latent_space_dim=args.latent_space_dim, proj_dim=args.proj_dim).to(device)
    
    supcon_loss_fn = SupervisedContrastiveLoss(temperature=args.temperature)

    lr = args.lr 
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': lr},
        {'params': decoder.parameters(), 'lr': lr},
        {'params': transform_head.parameters(), 'lr': lr}
    ], weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float('inf')
    no_improvement_epochs = 0

    if args.resume_from and os.path.isfile(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        transform_head.load_state_dict(checkpoint['transform_head_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        no_improvement_epochs = checkpoint.get('no_improvement_epochs', 0)
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(encoder, decoder, transform_head, train_dataloader, supcon_loss_fn, optimizer, device, args)
        val_loss = val_epoch(encoder, decoder, transform_head, valid_dataloader, supcon_loss_fn, device, args)

        print(f'EPOCH {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}')

        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.flush()
        wandb.log({"Epoch": epoch, "Loss/Train": train_loss, "Loss/Validation": val_loss})

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
        torch.save(checkpoint_dict, os.path.join(args.save_dir, 'miniSupCon_latest.pth'))
        if is_best:
            torch.save(checkpoint_dict, os.path.join(args.save_dir, 'miniSupCon_best.pth'))

        if no_improvement_epochs >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    writer.close()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE for Jet Anomaly Detection")
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=299)
    parser.add_argument('--latent_space_dim', type=int, default=128)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--patience', type=int, default=8)
    
    # Parametri Avanzati
    parser.add_argument('--beta', type=float, default=0.01, help='Peso della KL Divergence')
    parser.add_argument('--proj_dim', type=int, default=64, help='Dimensione output TransformHead')
    parser.add_argument('--temperature', type=float, default=0.1, help='Temperatura SupCon Loss')
    parser.add_argument('--contrastive_weight', type=float, default=1.0, help='Peso loss contrastiva')

    args = parser.parse_args()
    main(args)