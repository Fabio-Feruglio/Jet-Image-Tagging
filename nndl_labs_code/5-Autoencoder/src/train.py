import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataset import get_dataloaders
from model import Encoder, Decoder

### Training function
def train_epoch(encoder, decoder, device, dataloader, loss_fn, optimizer):
    # Set train mode for both the encoder and the decoder
    encoder.train()
    decoder.train()
    losses = []
    # Iterate the dataloader (we do not need the label values, this is unsupervised learning)
    for image_batch, _ in dataloader: # with "_" we just ignore the labels (the second element of the dataloader tuple)
        x_batch = image_batch.to(device)

        # Encode and decode data
        encoded_data = encoder(x_batch)
        decoded_data = decoder(encoded_data)

        # Evaluate loss
        loss = loss_fn(decoded_data, x_batch)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Record losses
        losses.append(loss.detach().cpu().numpy())
    losses = np.mean(losses)
    return losses


### Testing function
def test_epoch(encoder, decoder, device, dataloader, loss_fn):
    # Set evaluation mode for encoder and decoder
    encoder.eval()
    decoder.eval()

    with torch.no_grad(): # No need to track the gradients
        # Define the lists to store the outputs for each batch
        conc_out = []
        conc_label = []
        for image_batch, _ in dataloader:
            # Move tensor to the proper device
            image_batch = image_batch.to(device)
            # Encode data
            encoded_data = encoder(image_batch)
            # Decode data
            decoded_data = decoder(encoded_data)
            # Append the network output and the original image to the lists
            conc_out.append(decoded_data.cpu())
            conc_label.append(image_batch.cpu())
        # Create a single tensor with all the values in the lists
        conc_out = torch.cat(conc_out)
        conc_label = torch.cat(conc_label)
        # Evaluate global loss
        val_loss = loss_fn(conc_out, conc_label)
    return val_loss.data


def main(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Selected Device: {device}')
    
    # 2. Folders creation
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, 'images'), exist_ok=True)
    
    # 3. Load dataloaders
    train_loader, test_loader, test_dataset = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    
    # 4. Initialize models and loss function
    encoder = Encoder(encoded_space_dim=args.latent_dim).to(device)
    decoder = Decoder(encoded_space_dim=args.latent_dim).to(device)
    loss_fn = torch.nn.MSELoss()

    # 5. Define an optimizer (both for the encoder and the decoder!)
    lr = 5e-4 # Learning rate
    params_to_optimize = [
        {'params': encoder.parameters()},
        {'params': decoder.parameters()}
    ]
    optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-5)
    
    # 6. Training cycle
    for epoch in range(args.epochs):
        train_loss = train_epoch(encoder,decoder, device, train_loader, loss_fn, optimizer)
        val_loss = test_epoch(encoder, decoder, device, test_loader, loss_fn)
        
        print(f'EPOCH {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')
        
        # Saving test images
        img = test_dataset[0][0].unsqueeze(0).to(device)
        encoder.eval()
        decoder.eval()
        with torch.no_grad():
            encoded_img = encoder(img)
            rec_img = decoder(encoded_img)
            
        fig, axs = plt.subplots(1, 2, figsize=(12,6))
        axs[0].imshow(img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[0].set_title('Original')
        axs[1].imshow(rec_img.cpu().squeeze().numpy(), cmap='gist_gray')
        axs[1].set_title(f'Reconstructed (Epoch {epoch + 1})')
        plt.savefig(os.path.join(args.save_dir, 'images', f'epoch_{epoch+1}.jpg'))
        plt.close() # Close the figure!!
        
    # 7. Save weights
    encoder_path = os.path.join(args.save_dir, 'encoder_final.pth')
    decoder_path = os.path.join(args.save_dir, 'decoder_final.pth')
    torch.save(encoder.state_dict(), encoder_path)
    torch.save(decoder.state_dict(), decoder_path)
    print(f'Training completed. Models saved in {encoder_path}')

if __name__ == "__main__":
    # Command line args configuration
    parser = argparse.ArgumentParser(description="Train a simple Autoencoder on FashionMNIST")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch dimension')
    parser.add_argument('--lr', type=float, default=5e-4, help='Learning rate')
    parser.add_argument('--latent_dim', type=int, default=2, help='Latent space dimension')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory for data reading/saving')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory for model/results saving')
    
    args = parser.parse_args()
    main(args)