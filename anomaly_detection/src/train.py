import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

# Assicurati che i percorsi di import riflettano la tua repo!
from model.Autoencoder import Encoder, Decoder 
from dataset import JetH5Dataset

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    losses = []
    
    train_iterator = tqdm(dataloader, desc="Training")
    for image_batch, _ in train_iterator:
        image_batch = image_batch.to(device)
        
        # Forward pass
        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)
        
        # Loss computation
        loss = loss_fn(decoded_data, image_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_iterator.set_description(f"Train loss: {loss.item():.6f}")
        losses.append(loss.item())
        
    return np.mean(losses)

def val_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    losses = []
    
    with torch.no_grad():
        val_iterator = tqdm(dataloader, desc="Validation")
        for image_batch, _ in val_iterator:
            image_batch = image_batch.to(device)
            
            # Forward pass
            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)
            
            # Loss computation
            loss = loss_fn(decoded_data, image_batch)
            
            val_iterator.set_description(f"Val loss: {loss.item():.6f}")
            losses.append(loss.item())
            
    return np.mean(losses)

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Selected device: {device}')

    # Setup directory di salvataggio
    os.makedirs(args.save_dir, exist_ok=True)
    plot_dir = os.path.join(args.save_dir, f'autoencoder_progress_{args.encoded_space_dim}')
    os.makedirs(plot_dir, exist_ok=True)

    # Inizializzazione WandB
    wandb.init(
        project="jet-anomaly-detection",         # Nome del progetto su WandB
        name=f"ae_dim{args.encoded_space_dim}_lr{args.lr}", # Nome della singola run
        config=vars(args)                        # Salva tutti i parametri di argparse
    )

    # 1. Istanziamo i modelli
    encoder = Encoder(encoded_space_dim=args.encoded_space_dim).to(device)
    decoder = Decoder(encoded_space_dim=args.encoded_space_dim).to(device)

    # 2. Setup Ottimizzatore
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optim = torch.optim.Adam(params_to_optimize, lr=args.lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    # 3. Setup Dataset e DataLoader
    print(f"Caricamento dataset da: {args.data_path} ...")
    full_train_dataset = JetH5Dataset(args.data_path, mode='train')
    test_dataset = JetH5Dataset(args.data_path, mode='test')
    
    train_dataloader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Variabili per l'Early Stopping e Checkpointing
    best_val_loss = float('inf')
    no_improvement_epochs = 0
    patience = args.patience

    # 4. Training Loop
    for epoch in range(args.epochs):
        print(f'\nEPOCH {epoch + 1}/{args.epochs}')
        
        train_loss = train_epoch(encoder, decoder, device, train_dataloader, loss_fn, optim)
        val_loss = val_epoch(encoder, decoder, device, test_dataloader, loss_fn)
        
        print(f'TRAIN Loss: {train_loss:.6f} | VALIDATION Loss: {val_loss:.6f}')

        # --- Plot e Log Immagini ---
        img, label = test_dataset[0]
        img = img.unsqueeze(0).to(device)
        
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
            
        fig, axs = plt.subplots(1, 2, figsize=(10,5))
        axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[0].set_title(f'Original image (Label: {label.item()})')
        axs[0].axis('off')
        
        axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[1].set_title(f'Reconstructed (EPOCH {epoch + 1})')
        axs[1].axis('off')
        
        # Salva in locale
        save_path = os.path.join(plot_dir, f'epoch_{epoch + 1}.jpg')
        fig.savefig(save_path, bbox_inches='tight')
        
        # --- Log su WandB ---
        # Salviamo sia le metriche che l'immagine generata!
        wandb.log({
            "Epoch": epoch + 1,
            "Loss/Train": train_loss,
            "Loss/Validation": val_loss,
            "Reconstruction": wandb.Image(fig)
        })
        
        plt.close(fig)

        # --- Checkpointing & Early Stopping ---
        checkpoint_dict = {
            'epoch': epoch,
            'encoder_state_dict': encoder.state_dict(),
            'decoder_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optim.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improvement_epochs': no_improvement_epochs
        }

        # Salva sempre l'ultimo modello
        torch.save(checkpoint_dict, os.path.join(args.save_dir, 'autoencoder_latest.pth'))

        # Salva il best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_dict, os.path.join(args.save_dir, 'autoencoder_best.pth'))
            no_improvement_epochs = 0
            print("-> Nuovo Best Model salvato!")
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f'\nEarly stopping at epoch {epoch + 1}')
            break

    wandb.finish()
    print(f"\nTraining completato. Pesi salvati in {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Autoencoder for jet anomaly detection")
    parser.add_argument('--encoded_space_dim', type=int, default=128, help='Dimension of the latent space')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch dimension')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='/content/jet_images_299.h5', help='Path to the dataset file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_ae', help='Directory for model/results saving')
    parser.add_argument('--patience', type=int, default=10, help='Epochs to wait before early stopping')
    
    args = parser.parse_args()
    main(args)