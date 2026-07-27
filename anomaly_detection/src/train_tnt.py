import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset.dataloader_tnt import get_dataloaders_tnt
from model.other_models_attempt.autoencoder import Encoder, Decoder

def apply_pepper_noise(images, prob=0.1):
    if prob <= 0.0: return images
    mask = (torch.rand_like(images) > prob).float()
    return images * mask

def train_model(encoder, decoder, train_loader, val_loader, args, device, writer, save_dir, model_name="ae"):
    loss_fn = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=args.lr, weight_decay=args.weight_decay)
    
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\n=== Inizio addestramento {model_name} (Adam) ===")
    for epoch in range(args.epochs):
        encoder.train(); decoder.train()
        train_losses = []
        
        train_iterator = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x_batch, _ in train_iterator:
            x_batch = x_batch.to(device)
            x_noisy = apply_pepper_noise(x_batch, prob=args.noise_prob)
            
            reconstructed = decoder(encoder(x_noisy))
            loss = loss_fn(reconstructed, x_batch)
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_losses.append(loss.item())
            train_iterator.set_description(f"[{model_name}] Ep. {epoch+1} | Train Loss: {loss.item():.4f}")

        encoder.eval(); decoder.eval()
        val_losses = []
        with torch.no_grad():
            for x_batch, _ in val_loader:
                x_batch = x_batch.to(device)
                val_losses.append(loss_fn(decoder(encoder(x_batch)), x_batch).item())

        train_loss = np.mean(train_losses)
        val_loss = np.mean(val_losses)
        print(f"[{model_name}] Epoch {epoch+1} | Avg Train Loss: {train_loss:.4f} | Avg Val Loss: {val_loss:.4f}")

        writer.add_scalar(f'Loss/Train_{model_name}', train_loss, epoch)
        writer.add_scalar(f'Loss/Validation_{model_name}', val_loss, epoch)
        writer.flush()

        wandb.log({f"Epoch_{model_name}": epoch, f"Loss/Train_{model_name}": train_loss, f"Loss/Validation_{model_name}": val_loss})

        torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, os.path.join(save_dir, f'{model_name}_latest.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({'encoder': encoder.state_dict(), 'decoder': decoder.state_dict()}, os.path.join(save_dir, f'{model_name}_best.pth'))
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping {model_name} all'epoca {epoch+1}")
                break

def extract_pseudo_anomalies(encoder, decoder, tag_loader, device, percentile, batch_size):
    print(f"\n=== Estrazione Pseudo-Anomalie (Top {100-percentile:.1f}%) ===")
    encoder.eval(); decoder.eval()
    mse_fn = torch.nn.MSELoss(reduction='none')
    
    all_losses, all_labels = [], []
    
    # Memoria RAM Protetta: salviamo solo le Loss e gli Indici!
    with torch.no_grad():
        for x_batch, y_batch in tqdm(tag_loader, desc="Valutazione Tagging Set"):
            x_batch = x_batch.to(device)
            reconstructed = decoder(encoder(x_batch))
            loss_per_pixel = mse_fn(reconstructed, x_batch)
            loss_per_image = loss_per_pixel.view(loss_per_pixel.size(0), -1).mean(dim=1)
            
            all_losses.extend(loss_per_image.cpu().numpy())
            all_labels.extend(y_batch.numpy())

    all_losses = np.array(all_losses)
    all_labels = np.array(all_labels)
    threshold = np.percentile(all_losses, percentile)
    
    mask = all_losses > threshold
    indices_to_keep = np.where(mask)[0] # Otteniamo i numeri di riga da mantenere
    
    pseudo_labels = all_labels[mask]
    true_anomalies = np.sum(pseudo_labels == 1)
    print(f"-> Soglia MSE trovata: {threshold:.4f}")
    print(f"-> Eventi catturati (Dataset AE2): {len(indices_to_keep)}")
    print(f"-> Purezza vera nel set AE2: {true_anomalies}/{len(indices_to_keep)} ({true_anomalies/len(indices_to_keep)*100:.1f}%)")
    
    # IL MAGICO SUBSET: PyTorch crea un dataset virtuale senza clonare nulla in RAM!
    pseudo_dataset = Subset(tag_loader.dataset, indices_to_keep)
    
    val_size = max(1, int(0.15 * len(pseudo_dataset)))
    train_size = len(pseudo_dataset) - val_size
    train_ds, val_ds = random_split(pseudo_dataset, [train_size, val_size])
    
    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Selected Device: {device}')
    
    os.makedirs(args.save_dir_ae1, exist_ok=True)
    os.makedirs(args.save_dir_ae2, exist_ok=True)
    
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir_ae1, 'tensorboard_logs_tnt'))
    run = wandb.init(project="jet-tagging-anomaly-detection-ae-attempt", name=f"train_tnt_noise{args.noise_prob}_lr{args.lr}", config=vars(args))
    
    train_ae1_loader, val_ae1_loader, tag_loader, _ = get_dataloaders_tnt(
        args.data_path, args.bg_classes, args.img_size, args.batch_size, 0, args.num_train_samples, args.threshold_percentile
    )
    
    enc1 = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    dec1 = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    ae1_best_path = os.path.join(args.save_dir_ae1, 'ae1_bkg_best.pth')
    
    if args.skip_ae1 and os.path.exists(ae1_best_path):
        print(f"\n[!] Trovati pesi AE1 in: {ae1_best_path}. Salto addestramento AE1!")
    else:
        train_model(enc1, dec1, train_ae1_loader, val_ae1_loader, args, device, writer, args.save_dir_ae1, "ae1_bkg")
    
    checkpoint1 = torch.load(ae1_best_path, map_location=device, weights_only=False)
    enc1.load_state_dict(checkpoint1['encoder'])
    dec1.load_state_dict(checkpoint1['decoder'])
    
    train_ae2_loader, val_ae2_loader = extract_pseudo_anomalies(enc1, dec1, tag_loader, device, args.threshold_percentile, args.batch_size)
    
    enc2 = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    dec2 = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    train_model(enc2, dec2, train_ae2_loader, val_ae2_loader, args, device, writer, args.save_dir_ae2, "ae2_sig")
    
    writer.close(); wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--latent_space_dim', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--num_train_samples', type=int, default=30000)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--noise_prob', type=float, default=0.1)
    parser.add_argument('--threshold_percentile', type=float, default=90.0)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir_ae1', type=str, required=True)
    parser.add_argument('--save_dir_ae2', type=str, required=True)
    parser.add_argument('--skip_ae1', action='store_true')
    main(parser.parse_args())