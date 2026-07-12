import argparse
import os
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import wandb

from dataset.dataloader import get_dataloaders
from model.ensemble import EnsembleModel


### TRAINING ###
def train_epoch(model, dataloader, loss_fn, optimizer, device):
    model.train()
    losses = []
    accuracies = []

    train_iterator = tqdm(dataloader)
    for x_batch, label_batch in train_iterator:
        x_batch, label_batch = x_batch.to(device), label_batch.to(device)

        y_pred = model(x_batch) 
        loss = loss_fn(y_pred, label_batch) 

        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step()  

        train_iterator.set_description(f"Train loss: {loss.item():.4f}")
        losses.append(loss.item())

        pred = torch.argmax(y_pred, dim=1)
        acc = (pred == label_batch).float().mean()
        accuracies.append(acc.item())

    return np.mean(losses), np.mean(accuracies)


### VALIDATION ###
def val_epoch(model, dataloader, loss_fn, device):
    model.eval()
    losses = []
    accuracies = []

    with torch.no_grad():
        val_iterator = tqdm(dataloader)
        for x_batch, label_batch in val_iterator:
            x_batch, label_batch = x_batch.to(device), label_batch.to(device)

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
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f'Selected Device: {device}')
    
    # Folders creation
    os.makedirs(args.save_dir, exist_ok=True)
    #os.makedirs(os.path.join(args.save_dir, 'images'), exist_ok=True)
    # TensorBoard viewer setup
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard_logs'))

    # Wandb setup
    wandb.init(
        project="jet-tagging-main",             # Project name
        name=f"train_ensemble_lr{args.lr}",     # Name for the run
        config=vars(args)                       # Save  parameters
    )
    
    
    # Load dataloaders 
    train_dataloader, valid_dataloader, _ = get_dataloaders(data_filepath = args.data_path, 
                                                            img_size = args.img_size, batch_size = args.batch_size, 
                                                            num_workers = min(4, os.cpu_count() or 1),
                                                            max_samples = args.max_samples)
    
    model = EnsembleModel(num_classes = 5, 
                          resnet_path = args.resnet_weights, 
                          inception_path = args.inception_weights, 
                          device = str(device)).to(device)
    
    loss_fn = torch.nn.CrossEntropyLoss()

    # Freeze the backbone and train only final head for warmup epochs
    for param in model.resnet.parameters(): 
        param.requires_grad = False
    for param in model.inception.parameters(): 
        param.requires_grad = False

    optimizer = torch.optim.Adam(model.fc.parameters(), lr=args.lr)
    warmup_epochs = 3 
    warmup_stage = True

    best_val_loss = float('inf')
    patience = args.patience # For early stopping
    no_improvement_epochs = 0
    start_epoch = 0

    # Training resume logic
    if args.resume_from:
        if os.path.isfile(args.resume_from):
            print(f"Loading checkpoint from '{args.resume_from}' ...")
            checkpoint = torch.load(args.resume_from, map_location=device, weights_only=False)
            
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            if 'best_val_loss' in checkpoint:
                best_val_loss = checkpoint['best_val_loss']
            if 'no_improvement_epochs' in checkpoint:
                no_improvement_epochs = checkpoint['no_improvement_epochs']
            if 'warmup_stage' in checkpoint:
                warmup_stage = checkpoint['warmup_stage']
            print(f"Training resumed from {start_epoch}")
        else:
            print(f"No file found in '{args.resume_from}', starting from epoch = 0.")

    for epoch in range(start_epoch, args.epochs):
        
        # After warmup epochs, unfreeze the backbone for fine-tuning
        if warmup_stage and epoch == warmup_epochs:
            print("\n Warm-up completed... Now unfreezing the backbone for fine-tuning.")
            for param in model.resnet.parameters():
                param.requires_grad = True
            for param in model.inception.parameters():
                param.requires_grad = True
            
            # Different lr for backbone and final head
            optimizer = torch.optim.Adam([
                {'params': model.resnet.parameters(), 'lr': args.lr * 0.1},
                {'params': model.inception.parameters(), 'lr': args.lr * 0.1},
                {'params': model.fc.parameters(), 'lr': args.lr}
            ])
            warmup_stage = False

        print(f'\nEPOCH {epoch+1}/{args.epochs} [{args.mode.upper()}]')
        train_loss, train_acc = train_epoch(model, train_dataloader, loss_fn, optimizer, device)
        val_loss, val_acc = val_epoch(model, valid_dataloader, loss_fn, device)

        writer.add_scalar('Loss/Train_ensemble', train_loss, epoch)
        writer.add_scalar('Loss/Validation_ensemble', val_loss, epoch)
        writer.add_scalar('Accuracy/Train_ensemble', train_acc, epoch)
        writer.add_scalar('Accuracy/Validation_ensemble', val_acc, epoch)
        writer.flush()

        wandb.log({
            "Epoch": epoch,
            "Loss/Train": train_loss,
            "Loss/Validation": val_loss,
            "Accuracy/Train": train_acc,
            "Accuracy/Validation": val_acc
        })

        checkpoint_dict = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'no_improvement_epochs': no_improvement_epochs,
            'warmup_stage': warmup_stage
        }
        
        torch.save(checkpoint_dict, os.path.join(args.save_dir, 'ensemble_latest.pth'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint_dict, os.path.join(args.save_dir, 'ensemble_best.pth'))
            no_improvement_epochs = 0
        else:
            no_improvement_epochs += 1

        if no_improvement_epochs >= patience:
            print(f'Early stopping at epoch {epoch+1}')
            break

    writer.close()
    wandb.finish()
    print(f'Training completed. Best model saved in {os.path.join(args.save_dir, "ensemble_best.pth")}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Jet Image Classification Trainer")
    
    parser.add_argument('--resnet_weights', type=str, default=None, help='Resnet weights path')
    parser.add_argument('--inception_weights', type=str, default=None, help='Inception weights path')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=299)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to use for training")
    parser.add_argument('--patience', type=int, default=5, help='Number of epochs to wait for improvement before stopping')
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--resume_from', type=str, default=None)

    args = parser.parse_args()
    main(args)