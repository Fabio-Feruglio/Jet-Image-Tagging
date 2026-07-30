import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb
import torch.nn.functional as F

### CUSTOM LOSS FUNC FOR VAE
def VAE_loss_fn(reconstructed_x, x, mu, log_var, sigma=1.0):
    recon_loss = torch.nn.functional.mse_loss(reconstructed_x, x, reduction='mean') / (sigma**2)
    kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1).mean()
    num_pixels = x.shape[1] * x.shape[2] * x.shape[3]
    kl_div_scaled = kl_div / num_pixels
    return recon_loss + kl_div_scaled

### CUSTOM LOSS FUNC FOR HYBRID (SupCon)
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

### TRAINING ###
def train_epoch(encoder, decoder, dataloader, loss_fn, optimizer, device, args, supcon_loss_fn=None):
    encoder.train()
    decoder.train()
    losses = []

    train_iterator = tqdm(dataloader)
    for x_batch, label_batch in train_iterator:
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)

        # Forward pass
        if args.model == 'vae':
            encoded, mu, log_var = encoder(x_batch)
            reconstructed_x = decoder(encoded)
            loss = loss_fn(reconstructed_x, x_batch, mu, log_var)
            
        elif args.model == 'sae':
            encoded = encoder(x_batch)
            reconstructed_x = decoder(encoded)
            recon_loss = loss_fn(reconstructed_x, x_batch)  
            l1_penalty = torch.abs(encoded).sum(dim=1).mean()
            loss = recon_loss + args.l1_lambda * l1_penalty
            
        elif args.model == 'ae':
            encoded = encoder(x_batch)
            reconstructed_x = decoder(encoded)
            loss = loss_fn(reconstructed_x, x_batch)
            
        elif args.model == 'hybrid':
            z, p = encoder(x_batch)
            reconstructed_x = decoder(z)
            mse_loss = F.mse_loss(reconstructed_x, x_batch)
            supcon_loss = supcon_loss_fn(p, label_batch)
            loss = mse_loss + args.lambda_weight * supcon_loss

        # Backward pass
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Train loss: {loss.item():.4f}")
        losses.append(loss.item())

    avg_loss = np.mean(losses)
    return avg_loss

### VALIDATION ###
def val_epoch(encoder, decoder, dataloader, loss_fn, device, args, supcon_loss_fn=None):
    encoder.eval()
    decoder.eval()
    losses = []

    with torch.no_grad():
        val_iterator = tqdm(dataloader)

        for x_batch, label_batch in val_iterator:
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            if args.model == 'vae':
                encoded, mu, log_var = encoder(x_batch)
                reconstructed_x = decoder(encoded)
                loss = loss_fn(reconstructed_x, x_batch, mu, log_var)
                
            elif args.model == 'sae':
                encoded = encoder(x_batch)
                reconstructed_x = decoder(encoded)
                recon_loss = loss_fn(reconstructed_x, x_batch)
                l1_penalty = torch.abs(encoded).sum(dim=1).mean()
                loss = recon_loss + args.l1_lambda * l1_penalty
                
            elif args.model == 'ae':
                encoded = encoder(x_batch)
                reconstructed_x = decoder(encoded)
                loss = loss_fn(reconstructed_x, x_batch)
                
            elif args.model == 'hybrid':
                z, p = encoder(x_batch)
                reconstructed_x = decoder(z)
                mse_loss = F.mse_loss(reconstructed_x, x_batch)
                supcon_loss = supcon_loss_fn(p, label_batch)
                loss = mse_loss + args.lambda_weight * supcon_loss

            losses.append(loss.item())
            val_iterator.set_description(f"Val loss: {loss.item():.4f}")
            
    avg_loss = np.mean(losses)
    print(f"Validation Loss: {avg_loss:.4f}")
    return avg_loss

def main(args):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(f'Selected Device: {device}')
    
    # 2. Folders creation
    os.makedirs(args.save_dir, exist_ok=True)
    
    # --- SALVATAGGIO DEI PARAMETRI IN UN FILE DI TESTO ---
    config_path = os.path.join(args.save_dir, f"{args.model}_training_config.txt")
    with open(config_path, 'w') as f:
        f.write("--- TRAINING CONFIGURATION ---\n")
        for k, v in vars(args).items():
            f.write(f"{k}: {v}\n")
    print(f"Training configuration saved to: {config_path}")

    # TensorBoard viewer setup
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, f'tensorboard_logs_{args.model}'))

    wandb_run_id = None
    if args.resume_from and os.path.isfile(args.resume_from):
        temp_checkpoint = torch.load(args.resume_from, map_location='cpu', weights_only=False)
        if 'wandb_run_id' in temp_checkpoint:
            wandb_run_id = temp_checkpoint['wandb_run_id']

    run_name = f"train_{args.model}_lr{args.lr}_{args.optimizer}"
    if args.model == 'sae':
        run_name += f"_l1{args.l1_lambda}"
    elif args.model == 'hybrid':
        run_name += f"_lambda{args.lambda_weight}"
        
    run = wandb.init(
        project = f"jet-tagging-anomaly-detection-{args.model}-attempt",             
        name = run_name,                    
        config = vars(args),
        id = wandb_run_id,     
        resume = "allow"                                     
    )
    
    # 3. Load dataloaders 
    if args.model == 'hybrid':
        from dataset.dataloader_supcon import get_dataloaders
        train_dataloader, valid_dataloader, _ = get_dataloaders(
            data_filepath = args.data_path, 
            bg_classes = args.bg_classes,
            img_size = args.img_size, 
            batch_size = args.batch_size, 
            num_workers = min(4, os.cpu_count() or 1),
            max_samples = args.max_samples,
            binary_train_labels=False
        )
    else:
        from dataset.dataloader import get_dataloaders
        train_dataloader, valid_dataloader, _ = get_dataloaders(
            data_filepath = args.data_path, 
            bg_classes = args.bg_classes,
            img_size = args.img_size, 
            batch_size = args.batch_size, 
            num_workers = min(4, os.cpu_count() or 1),
            max_samples = args.max_samples
        )
    
    # 4. Initialize model and loss function
    if args.model == 'vae':
        from anomaly_detection.src.model.miniVAE import Encoder, Decoder
        encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    elif args.model == 'hybrid':
        from model.autoencoder import HybridEncoder as Encoder, Decoder
        encoder = Encoder(latent_space_dim=args.latent_space_dim, proj_dim=64).to(device)
    else:
        from model.autoencoder import Encoder, Decoder
        encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
        
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    
    supcon_loss_fn = None
    if args.model == 'vae':
        loss_fn = VAE_loss_fn
    elif args.model == 'hybrid':
        loss_fn = None # Handled explicitly in train_epoch
        supcon_loss_fn = SupConLoss(temperature=0.1)
    else:
        loss_fn = torch.nn.MSELoss()

    # 5. Define an optimizer 
    if args.optimizer.lower() == 'adagrad':
        optimizer = torch.optim.Adagrad([
            {'params': encoder.parameters(), 'lr': args.lr},
            {'params': decoder.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)
    else:
        optimizer = torch.optim.Adam([
            {'params': encoder.parameters(), 'lr': args.lr},
            {'params': decoder.parameters(), 'lr': args.lr}
        ], weight_decay=args.weight_decay)

    start_epoch = 0
    best_val_loss = float('inf')
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
        else:
            print(f"No file found in '{args.resume_from}', starting from epoch = 0.")
    
    # 6. Training cycle
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(encoder, decoder, train_dataloader, loss_fn, optimizer, device, args, supcon_loss_fn)
        val_loss = val_epoch(encoder, decoder, valid_dataloader, loss_fn, device, args, supcon_loss_fn)

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
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improvement_epochs': no_improvement_epochs,
            'wandb_run_id': run.id,
            'config': vars(args) 
        }
        
        torch.save(checkpoint_dict, os.path.join(args.save_dir, f'{args.model}_latest.pth'))
        if is_best:
            torch.save(checkpoint_dict, os.path.join(args.save_dir, f'{args.model}_best.pth'))

        if no_improvement_epochs >= args.patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    writer.close()
    wandb.finish()
    print(f'Training completed. Best model saved in {os.path.join(args.save_dir, f"{args.model}_best.pth")}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Train script for AE, VAE, SAE, Hybrid")
    parser.add_argument('--model', type=str, required=True, choices=['ae', 'vae', 'sae', 'hybrid'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'adagrad'], help='Choice of optimizer')
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--latent_space_dim', type=int, default=128)
    parser.add_argument('--l1_lambda', type=float, default=1e-4)
    parser.add_argument('--lambda_weight', type=float, default=0.1, help='Weight for SupCon loss in Hybrid model')
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--patience', type=int, default=5)
    args = parser.parse_args()
    main(args)
