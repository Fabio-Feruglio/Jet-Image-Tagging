import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.autoencoder import Encoder, Decoder

def apply_spatial_dropout(x, drop_prob=0.2):
    mask = (torch.rand_like(x) > drop_prob).float()
    return x * mask

def weighted_mse_loss(reconstructed_x, original_x, active_weight=5.0):
    mse_per_pixel = F.mse_loss(reconstructed_x, original_x, reduction='none')
    active_mask = (original_x > 0).float()
    weights = 1.0 + (active_weight - 1.0) * active_mask
    return (mse_per_pixel * weights).mean()

def train_epoch(encoder, decoder, dataloader, optimizer, device, active_weight, noise_factor):
    encoder.train()
    decoder.train()
    losses = []
    train_iterator = tqdm(dataloader)
    for x_batch, _ in train_iterator:
        x_batch = x_batch.to(device)
        x_corrupted = apply_spatial_dropout(x_batch, drop_prob=noise_factor)
        
        encoded = encoder(x_corrupted)
        reconstructed_x = decoder(encoded)
        loss = weighted_mse_loss(reconstructed_x, x_batch, active_weight=active_weight) 

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Train loss: {loss.item():.4f}")
        losses.append(loss.item())
    return np.mean(losses)

def val_epoch(encoder, decoder, dataloader, device, active_weight):
    encoder.eval()
    decoder.eval()
    losses = []
    with torch.no_grad():
        val_iterator = tqdm(dataloader)
        for x_batch, _ in val_iterator:
            x_batch = x_batch.to(device)
            # In validation NO rumore
            encoded = encoder(x_batch)
            reconstructed_x = decoder(encoded)
            loss = weighted_mse_loss(reconstructed_x, x_batch, active_weight=active_weight)
            losses.append(loss.item())
            val_iterator.set_description(f"Val loss: {loss.item():.4f}")
    return np.mean(losses)

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
            print(f"ID: {wandb_run_id}")

    run = wandb.init(
        project="jet-tagging-anomaly-detection", 
        name=f"train_dae_lr{args.lr}", 
        config=vars(args), 
        id=wandb_run_id, 
        resume="allow"
    )
    
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        data_filepath=args.data_path, 
        bg_classes=args.bg_classes,
        img_size=args.img_size, 
        batch_size=args.batch_size, 
        num_workers=min(4, os.cpu_count() or 1), 
        max_samples=args.max_samples
    )
    
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': args.lr},
        {'params': decoder.parameters(), 'lr': args.lr}
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
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            if 'no_improvement_epochs' in checkpoint:
                no_improvement_epochs = checkpoint['no_improvement_epochs']
            print(f"Training resumed from {start_epoch}")
        else:
            print(f"No file found in '{args.resume_from}', starting from epoch = 0.")
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(encoder, decoder, train_dataloader, optimizer, device, args.active_weight, args.noise_factor)
        val_loss = val_epoch(encoder, decoder, valid_dataloader, device, args.active_weight)

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
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improvement_epochs': no_improvement_epochs,
            'wandb_run_id': run.id
        }
        torch.save(checkpoint_dict, os.path.join(args.save_dir, 'dae_latest.pth'))
        if is_best:
            torch.save(checkpoint_dict, os.path.join(args.save_dir, 'dae_best.pth'))

        if no_improvement_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    writer.close()
    wandb.finish()
    print(f'Training completed. Best model saved in {os.path.join(args.save_dir, "dae_best.pth")}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--latent_space_dim', type=int, default=128)
    parser.add_argument('--active_weight', type=float, default=5.0)
    parser.add_argument('--noise_factor', type=float, default=0.2)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    main(args)