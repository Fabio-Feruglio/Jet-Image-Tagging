import os
import argparse
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

# Assicurati che i percorsi di import riflettano la tua repo!
from model.Autoencoder import Encoder, Decoder 
from dataset import JetH5Dataset

def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    encoder.train()
    decoder.train()
    losses = []
    
    for image_batch, _ in dataloader:
        image_batch = image_batch.to(device)
        encoded_data = encoder(image_batch)
        decoded_data = decoder(encoded_data)
        
        loss = loss_fn(decoded_data, image_batch)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
    return np.mean(losses)

def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    encoder.eval()
    decoder.eval()
    losses = []
    
    with torch.no_grad():
        for image_batch, _ in dataloader:
            image_batch = image_batch.to(device)
            encoded_data = encoder(image_batch)
            decoded_data = decoder(encoded_data)
            
            # Calcoliamo la loss batch per batch, senza accumulare i tensori
            loss = loss_fn(decoded_data, image_batch)
            losses.append(loss.item())
            
    return np.mean(losses)

def main(args):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # Setup directory di salvataggio
    os.makedirs(args.save_dir, exist_ok=True)
    plot_dir = os.path.join(args.save_dir, f'autoencoder_progress_{args.encoded_space_dim}')
    os.makedirs(plot_dir, exist_ok=True)

    # 1. Istanziamo i modelli usando i parametri di argparse
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
    # Train solo su background
    full_train_dataset = JetH5Dataset(args.data_path, mode='train')
    
    # Validation su un mix di tutto (per vedere se la loss sale per le anomalie)
    test_dataset = JetH5Dataset(args.data_path, mode='test')
    
    train_dataloader = DataLoader(full_train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # 4. Training Loop
    for epoch in range(args.epochs):
        print(f'EPOCH {epoch + 1}/{args.epochs}')
        
        train_loss = train_epoch(encoder, decoder, device, train_dataloader, loss_fn, optim)
        print(f'TRAIN - loss: {train_loss:.6f}')
        
        val_loss = test_epoch(encoder, decoder, device, test_dataloader, loss_fn)
        print(f'VALIDATION (Mix Background/Anomaly) - loss: {val_loss:.6f}\n')

        # Plot progress (prende la prima immagine del test_dataset)
        img, label = test_dataset[0]
        img = img.unsqueeze(0).to(device)
        
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            rec_img = decoder(encoder(img))
            
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[0].set_title(f'Original image (Label: {label.item()})')
        axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[1].set_title(f'Reconstructed (EPOCH {epoch + 1})')
        
        save_path = os.path.join(plot_dir, f'epoch_{epoch + 1}.jpg')
        fig.savefig(save_path)
        plt.close(fig) # Chiudi la figura per non intasare Colab

    # Save network parameters nella save_dir specificata
    torch.save(encoder.state_dict(), os.path.join(args.save_dir, 'encoder_params.pth'))
    torch.save(decoder.state_dict(), os.path.join(args.save_dir, 'decoder_params.pth'))
    print(f"Training completato. Pesi salvati in {args.save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the Autoencoder for jet anomaly detection")
    parser.add_argument('--encoded_space_dim', type=int, default=128, help='Dimension of the latent space')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch dimension')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--data_path', type=str, default='/content/jet_images_299.h5', help='Path to the dataset file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints_ae', help='Directory for model/results saving')
    
    args = parser.parse_args()
    main(args)