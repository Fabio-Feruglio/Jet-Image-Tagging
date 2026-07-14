import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

# Assicurati che i percorsi di import riflettano la tua repo!
from model.Autoencoder import Encoder, Decoder 
from dataset import JetH5Dataset

# --- HYPERPARAMETERS ---
encoded_space_dim = 128
batch_size = 64
lr = 5e-4
num_epochs = 10
# Puntiamo al file copiato localmente in /content/ per massima velocità
dataset_path = '/content/jet_images_299.h5' 

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

def main():
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f'Selected device: {device}')

    # 1. Istanziamo i modelli
    encoder = Encoder(encoded_space_dim=encoded_space_dim).to(device)
    decoder = Decoder(encoded_space_dim=encoded_space_dim).to(device)

    # 2. Setup Ottimizzatore
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optim = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()

    # 3. Setup Dataset e DataLoader
    print("Caricamento dataset...")
    # Train solo su background
    full_train_dataset = JetH5Dataset(dataset_path, mode='train')
    
    # Validation su un mix di tutto (per vedere se la loss sale per le anomalie)
    test_dataset = JetH5Dataset(dataset_path, mode='test')

    # Split opzionale del train dataset per validation "pura" sul background
    # Per semplicità qui usiamo train su q,g e test su tutto.
    
    train_dataloader = DataLoader(full_train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 4. Training Loop
    os.makedirs(f'autoencoder_progress_{encoded_space_dim}_features', exist_ok=True)

    for epoch in range(num_epochs):
        print(f'EPOCH {epoch + 1}/{num_epochs}')
        
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
        
        save_path = f'autoencoder_progress_{encoded_space_dim}_features/epoch_{epoch + 1}.jpg'
        fig.savefig(save_path)
        plt.close(fig) # Chiudi la figura per non intasare Colab

    # Save network parameters
    torch.save(encoder.state_dict(), 'encoder_params.pth')
    torch.save(decoder.state_dict(), 'decoder_params.pth')
    print("Training completato e parametri salvati.")

if __name__ == "__main__":
    main()