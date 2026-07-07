import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

from dataset.dataloader import get_dataloaders
from model.resnet import ResNet50
from model.inception import InceptionV4

### TRAINING ###
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []
    accuracies = []

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

        train_iterator.set_description(f"Train loss: {loss.item():.4f}")
        losses.append(loss.item())

        #train_iterator.set_description(f"Train loss: {loss.detach().cpu().numpy()}")
        #losses.append(loss.detach().cpu().numpy())

        # Accuracy computation (we have a 5-class classification problem)
        pred = torch.argmax(y_pred, dim=1)
        acc = (pred == label_batch).float().mean()
        accuracies.append(acc.item())

    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    return avg_loss, avg_acc


### VALIDATION ###
def val_epoch(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    accuracies = []

    with torch.no_grad():
        val_iterator = tqdm(dataloader)

        for x_batch, label_batch in val_iterator:
            x_batch = x_batch.to(device)
            label_batch = label_batch.to(device)

            y_pred = model(x_batch)
            loss = loss_fn(y_pred, label_batch)

            pred = torch.argmax(y_pred, dim=1)
            acc = (pred == label_batch).float().mean()
            losses.append(loss.item())
            accuracies.append(acc.item())
            
            val_iterator.set_description(f"Val loss: {loss.item():.4f}")
            
    avg_loss = np.mean(losses)
    avg_acc = np.mean(accuracies)
    
    print(f"Validation Loss: {avg_loss:.4f} | Validation Accuracy: {avg_acc:.4f}")
    return avg_loss, avg_acc


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
    train_dataloader, valid_dataloader, _ = get_dataloaders(data_filepath = args.data_path, 
                                                            img_size = args.img_size, batch_size = args.batch_size, 
                                                            num_workers = min(4, os.cpu_count() or 1),
                                                            max_samples = args.max_samples)
    
    # 4. Initialize model and loss function
    if args.mode == 'resnet':
        model = ResNet50().to(device)
    elif args.mode == 'inception':
        model = InceptionV4().to(device)
    else:
        raise ValueError("Non-supported mode. Please choose 'resnet' or 'inception'.")
    loss_fn = torch.nn.CrossEntropyLoss()

    # 5. Define an optimizer 
    lr = args.lr # Learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,  weight_decay=1e-4)

    start_epoch = 0
    best_val_loss = float('inf')
    patience = args.patience # For early stopping
    no_improvement_epochs = 0

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
            if 'no_improvement_epochs' in checkpoint:
                no_improvement_epochs = checkpoint['no_improvement_epochs']
            print(f"Training resumed from {start_epoch}")
        else:
            print(f"No file found in '{args.resume_from}', starting from epoch = 0.")
    
    # 6. Training cycle
    for epoch in range(start_epoch, args.epochs):
        train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = val_epoch(model, valid_dataloader, loss_fn, device)

        print(f'EPOCH {epoch+1}/{args.epochs} - Train Loss: {train_loss:.4f} - Train Accuracy: {train_acc:.4f}')
        print(f'EPOCH {epoch+1}/{args.epochs} - Validation Loss: {val_loss:.4f} - Validation Accuracy: {val_acc:.4f}')

        # Add values for TensorBoard viewer
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Validation', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation', val_acc, epoch)
        writer.flush()

        # Save the checkpoint
        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improvement_epochs': no_improvement_epochs
        }
        
        # Save the best model
        torch.save(checkpoint_dict, os.path.join(args.save_dir, f'{args.mode}_latest.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_dict, os.path.join(args.save_dir, f'{args.mode}_best.pth'))
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        # Early stopping
        if no_improvement_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    writer.close()
    print(f'Training completed. Best model saved in {os.path.join(args.save_dir, f"{args.mode}_best.pth")}')

if __name__ == "__main__":
    # Command line args configuration
    parser = argparse.ArgumentParser(description="Train the ensemble model for jet image classification")
    parser.add_argument('--mode', type=str, default='resnet', choices=['resnet', 'inception'], help="Choose the model to train: 'resnet' or 'inception'")
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=256, help='Batch dimension')
    parser.add_argument('--img_size', type=int, default=299, help='Image size for resizing')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to use for training")
    parser.add_argument('--data_path', type=str, default='./dataset.h5', help='Path to the dataset file')
    parser.add_argument('--save_dir', type=str, default='./checkpoints', help='Directory for model/results saving')
    parser.add_argument('--resume_from', type=str, default=None, help="Path to weights already trained to resume training")
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for improvement before stopping')

    args = parser.parse_args()
    main(args)