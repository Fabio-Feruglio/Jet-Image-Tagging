import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.miniVAE import Encoder, Decoder

###CUSTOM LOSS FUNC FOR VAE
def VAE_loss_fn(reconstructed_x, x, mu, log_var, sigma=1.0):
    # 1. MSE puro (Media su tutto il batch e tutti i pixel)
    # Valore atteso all'inizio: circa 0.05 - 0.2
    recon_loss = torch.nn.functional.mse_loss(reconstructed_x, x, reduction='mean') / (sigma**2)

    # 2. KL Divergence (Media sul batch)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()

    # 3. Dividiamo la KL per il numero di pixel per bilanciarla con la media della MSE
    num_pixels = x.shape[1] * x.shape[2] * x.shape[3]
    print(f"Number of pixels: {num_pixels}")
    print(f"x shape: {x.shape}")
    kl_div_scaled = kl_div / num_pixels

    # Valore finale piccolo e stabile
    return recon_loss + kl_div_scaled

### TRAINING ###
def train_epoch(encoder, decoder, dataloader, loss_fn, optimizer, device):
    encoder.train()
    decoder.train()
    losses = []

    train_iterator = tqdm(dataloader)
    for x_batch, label_batch in train_iterator:
        x_batch = x_batch.to(device)

        if x_batch.max() > 1.0:
            x_batch = x_batch / 255.0

        label_batch = label_batch.to(device)

        # Forward pass
        encoded, mu, log_var = encoder(x_batch)
        reconstructed_x = decoder(encoded)

        # Loss computation
        loss = loss_fn(reconstructed_x, x_batch, mu, log_var) 

        # Backward pass
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Train loss: {loss.item():.4f}")
        losses.append(loss.item())

    avg_loss = np.mean(losses)

    return avg_loss


### VALIDATION ###
def val_epoch(encoder, decoder, dataloader, loss_fn, device):
    encoder.eval()
    decoder.eval()
    losses = []

    with torch.no_grad():
        val_iterator = tqdm(dataloader)

        for x_batch, label_batch in val_iterator:
            x_batch = x_batch.to(device)

            if x_batch.max() > 1.0:
                x_batch = x_batch / 255.0
            
            label_batch = label_batch.to(device)

            encoded, mu, log_var = encoder(x_batch)
            reconstructed_x = decoder(encoded)
            loss = loss_fn(reconstructed_x, x_batch, mu, log_var)

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

    # TensorBoard viewer setup
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard_logs_anomaly_detection'))

    wandb_run_id = None
    if args.resume_from and os.path.isfile(args.resume_from):
        
        temp_checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        if 'wandb_run_id' in temp_checkpoint:
            wandb_run_id = temp_checkpoint['wandb_run_id']
            print(f"ID: {wandb_run_id}")

    # Wandb setup
    run = wandb.init(
        project = "jet-tagging-anomaly-detection-vae-attempt",             # Project name
        name = f"train_vae_lr{args.lr}",                    # Name for the run
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
    
    # 4. Initialize model and loss function
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    loss_fn = VAE_loss_fn

    # 5. Define an optimizer 
    lr = args.lr 
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': lr},
        {'params': decoder.parameters(), 'lr': lr}
    ], weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float('inf')
    patience = args.patience 
    no_improvement_epochs = 0

    # Training resume logic
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
    
    # 6. Training cycle
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(encoder, decoder, train_dataloader, loss_fn, optimizer, device)
        val_loss = val_epoch(encoder, decoder, valid_dataloader, loss_fn, device)

        print(f'EPOCH {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f}')
        print(f'EPOCH {epoch+1}/{args.epochs} - Validation Loss: {val_loss:.4f}')

        # Add values for TensorBoard viewer
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.flush()

        # Add values for WandB logger
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
    # Command line args configuration
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

    args = parser.parse_args()
    main(args)