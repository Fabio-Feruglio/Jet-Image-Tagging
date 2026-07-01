import argparse
import os
import torch
import numpy as np
from dataset import get_dataloaders
from model import InceptionV4
from tqdm import tqdm # type: ignore
from torch.utils.tensorboard import SummaryWriter

### TRAINING ###
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []

    train_iterator = tqdm(dataloader)
    for x_batch, label_batch in train_iterator:
        x_batch = x_batch.to(device)
        label_batch = label_batch.to(device)

        # Forward pass
        y_pred = model(x_batch) 

        # Loss computation
        loss = loss_fn(y_pred, label_batch) 

        # Backward pass
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Train loss: {loss.detach().cpu().numpy()}")
        losses.append(loss.detach().cpu().numpy())
    losses = np.mean(losses)
    return losses


### VALIDATION ###
def val_epoch(model, dataloader, loss_fn, device):
    model.eval()
    with torch.no_grad():
        predictions = []
        true = []
        val_iterator = tqdm(dataloader)
        for batch_x, batch_y in val_iterator:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            predictions.append(y_pred.cpu())
            true.append(batch_y.cpu())
            
        predictions = torch.cat(predictions, dim=0)
        true = torch.cat(true, dim=0)
        val_loss = loss_fn(predictions, true)
        val_acc = (torch.sigmoid(predictions).round() == true).float().mean()
        print(f"loss: {val_loss}, accuracy: {val_acc}")

    return val_loss


def main(args):
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Selected Device: {device}')
    
    # 2. Folders creation
    os.makedirs(args.save_dir, exist_ok=True)
    #os.makedirs(os.path.join(args.save_dir, 'images'), exist_ok=True)
    # TensorBoard viewer setup
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard_logs'))
    
    
    # 3. Load dataloaders 
    train_dataloader, valid_dataloader, test_dataloader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size, max_train_samples=32000)
    
    # 4. Initialize model and loss function
    model = InceptionV4().to(device)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    # 5. Define an optimizer 
    lr = args.lr # Learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay = 0)

    start_epoch = 0
    best_val_loss = float('inf')

    # Training resume logic
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint from '{args.resume_from}' ...")
            checkpoint = torch.load(args.resume_from, map_location=device)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            
            print(f"Training resumed from {start_epoch}")
        else:
            print(f"No file found in '{args.resume_from}', starting from epoch = 0.")
    
    # 6. Training cycle
    for epoch in range(start_epoch, args.epochs):
        train_loss = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        val_loss = val_epoch(model, valid_dataloader, loss_fn, device)

        if epoch % 100 == 0:
            print(f'EPOCH {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Val Loss: {val_loss:.4f}')

        # Add values for TensorBoard viewer
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)

        # Save the checkpoint
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss
        }
        torch.save(checkpoint_dict, os.path.join(args.save_dir, 'latest_checkpoint.pth'))

        # Save the best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_dict, os.path.join(args.save_dir, 'best_model.pth'))
            
        
    writer.close()
    print(f'Training completed. Best model saved in {os.path.join(args.save_dir, 'best_model.pth')}')

if __name__ == "__main__":
    # Command line args configuration
    parser = argparse.ArgumentParser(description="Train a simple Autoencoder on FashionMNIST")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch dimension')
    parser.add_argument('--lr', type=float, default=1e-2, help='Learning rate')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory for data reading/saving')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory for model/results saving')
    parser.add_argument('--resume_from', type=str, default=None, help="Path to weights alrady trained to resume training")
    
    args = parser.parse_args()
    main(args)