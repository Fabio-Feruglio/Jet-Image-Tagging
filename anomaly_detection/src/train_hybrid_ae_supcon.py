import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.hybrid_ae_supcon import HybridEncoder, Decoder

class SupConLoss(torch.nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    def forward(self, features, labels):
        device, batch_size = features.device, features.shape[0]
        sim_matrix = torch.matmul(features, features.T) / self.temperature
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        sim_max, _ = torch.max(sim_matrix, dim=1, keepdim=True)
        logits = sim_matrix - sim_max.detach()
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-9)
        mask_sum = mask.sum(1)
        mask_sum = torch.where(mask_sum == 0, torch.ones_like(mask_sum), mask_sum)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_sum
        return -mean_log_prob_pos.mean()

def train_epoch(encoder, decoder, dataloader, optimizer, device, supcon_loss_fn, lambda_weight):
    encoder.train(); decoder.train()
    losses = []
    train_iterator = tqdm(dataloader)
    for x_batch, labels in train_iterator:
        x_batch, labels = x_batch.to(device), labels.to(device)
        z, p = encoder(x_batch)
        reconstructed_x = decoder(z)

        mse_loss = F.mse_loss(reconstructed_x, x_batch)
        supcon_loss = supcon_loss_fn(p, labels)
        loss = mse_loss + lambda_weight * supcon_loss

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Train loss: {loss.item():.4f}")
        losses.append(loss.item())
    return np.mean(losses)

def val_epoch(encoder, decoder, dataloader, device, supcon_loss_fn, lambda_weight):
    encoder.eval(); decoder.eval()
    losses = []
    with torch.no_grad():
        val_iterator = tqdm(dataloader)
        for x_batch, labels in val_iterator:
            x_batch, labels = x_batch.to(device), labels.to(device)
            z, p = encoder(x_batch)
            reconstructed_x = decoder(z)
            mse_loss = F.mse_loss(reconstructed_x, x_batch)
            # Calcoliamo supcon_loss anche in validation se le label sono note (background)
            # NB: Se ci sono anomalie (classe > 1), SupCon le scarta, ma su valid_loader ci sono solo bg.
            supcon_loss = supcon_loss_fn(p, labels)
            loss = mse_loss + lambda_weight * supcon_loss
            losses.append(loss.item())
            val_iterator.set_description(f"Val loss: {loss.item():.4f}")
    return np.mean(losses)

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    os.makedirs(args.save_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard_logs'))

    wandb_run_id = None
    if args.resume_from and os.path.isfile(args.resume_from):
        temp_checkpoint = torch.load(args.resume_from, map_location='cpu')
        wandb_run_id = temp_checkpoint.get('wandb_run_id', None)

    run = wandb.init(project="jet-tagging-anomaly-detection", name=f"train_hybrid_lr{args.lr}", config=vars(args), id=wandb_run_id, resume="allow")
    
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        data_filepath=args.data_path, bg_classes=args.bg_classes,
        img_size=args.img_size, batch_size=args.batch_size, 
        num_workers=min(4, os.cpu_count() or 1), max_samples=args.max_samples
    )
    
    encoder = HybridEncoder(latent_space_dim=args.latent_space_dim, proj_dim=64).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    supcon_loss_fn = SupConLoss(temperature=0.1)

    start_epoch, no_improvement_epochs = 0, 0
    best_val_loss = float('inf')

    if args.resume_from and os.path.isfile(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        no_improvement_epochs = checkpoint.get('no_improvement_epochs', 0)
    
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(encoder, decoder, train_dataloader, optimizer, device, supcon_loss_fn, args.lambda_weight)
        val_loss = val_epoch(encoder, decoder, valid_dataloader, device, supcon_loss_fn, args.lambda_weight)

        print(f'EPOCH {epoch+1}/{args.epochs} - Train: {train_loss:.4f} | Val: {val_loss:.4f}')
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        wandb.log({"Epoch": epoch, "Loss/Train": train_loss, "Loss/Validation": val_loss})

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        checkpoint_dict = {
            'epoch': epoch, 'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss, 'no_improvement_epochs': no_improvement_epochs, 'wandb_run_id': run.id
        }
        torch.save(checkpoint_dict, os.path.join(args.save_dir, 'hybrid_latest.pth'))
        if is_best:
            torch.save(checkpoint_dict, os.path.join(args.save_dir, 'hybrid_best.pth'))

        if no_improvement_epochs >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    writer.close()
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--latent_space_dim', type=int, default=128)
    parser.add_argument('--lambda_weight', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    main(args)