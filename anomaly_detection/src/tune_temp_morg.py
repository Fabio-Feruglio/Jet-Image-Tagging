import argparse
import os
from tqdm import tqdm
import optuna
import wandb
import torch
import torch.nn as nn
import numpy as np

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.autoencoder import Encoder, Decoder


def objective(trial, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    latent_space_dim = trial.suggest_categorical("latent_space_dim", [64, 128, 256, 512])
    lr = trial.suggest_categorical("lr_mlp", [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2])                            
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])    # Weight decay
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])                         # Batch size

    
    print(f"\nTRIAL {trial.number}\n{'='*40}")
    print("Hyperparameters:")
    print(f"Latent Space Dim: {latent_space_dim}")
    print(f"Learning Rate: {lr:.6f}")
    print(f"Weight Decay: {weight_decay:.6f}")
    print(f"Batch size: {batch_size}")
    
    wandb.init(
        project = "jet-tagging-ae-tuning",
        group = "optuna-ae-ensemble",
        name = f"trial_ae_{trial.number}",
        config = trial.params,
        reinit = True
    )
    
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        data_filepath = args.data_path, 
        img_size = args.img_size, 
        batch_size = batch_size, 
        num_workers = min(4, os.cpu_count() or 1),
        max_samples = args.max_samples
    )
    
    encoder = Encoder(latent_space_dim=latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=latent_space_dim).to(device)
    
    # Optimizer separato per sezioni del VAE
    optimizer = torch.optim.Adam([
        {'params': encoder.parameters(), 'lr': lr},
        {'params': decoder.parameters(), 'lr': lr}
    ], weight_decay = weight_decay)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None # type: ignore

    loss_fn = nn.MSELoss()

    best_val_loss = float('inf')

    for epoch in range(args.tune_epochs):
        
        encoder.train()
        decoder.train()

        train_losses = []
        print(f"\nEPOCH {epoch+1}/{args.tune_epochs}")

        ### TRAINING LOOP
        for batch_x, _ in tqdm(train_dataloader, desc = "Training"):
            
            batch_x = batch_x.to(device)
            
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'): #type: ignore
                reconstructed_x = decoder(encoder(batch_x))
                loss = loss_fn(reconstructed_x, batch_x)
                
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
            
        print(f"Train Loss: {np.mean(train_losses):.4f}")

        ### VALIDATION LOOP
        encoder.eval()
        decoder.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in tqdm(valid_dataloader, desc = "Validation"):
                batch_x = batch_x.to(device)
                
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'): #type: ignore
                    reconstructed_x = decoder(encoder(batch_x))
                    loss = loss_fn(reconstructed_x, batch_x)
                
                val_losses.append(loss.item())
                
        print(f"Val Loss: {np.mean(val_losses):.4f}")

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
        })
        
        # Optuna Pruning
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            wandb.finish() 
            print(f"Trial pruned at epoch {epoch+1} with val loss {epoch_val_loss:.4f}")
            raise optuna.exceptions.TrialPruned()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
    
    print(f"Trial {trial.number} completed with best val loss: {best_val_loss:.4f}")
    wandb.finish()
    
    return float(best_val_loss)

def main(args):
    print("Optimization of AE hyperparameters...")

    os.makedirs(args.save_dir, exist_ok=True)
    db_path = os.path.join(args.save_dir, "optuna_study_ae.db")
    storage_name = f"sqlite:///{db_path}"
    study_name = "jet_tagging_study_ae"
    
    study = optuna.create_study(
        study_name = study_name,
        storage = storage_name,
        load_if_exists = True,
        direction = "minimize", 
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=args.warmup_epochs)
    )
    
    print(f"Already completed trials: {len(study.trials)}")
    study.optimize(lambda trial: objective(trial, args), n_trials = args.n_trials)
    
    print("The best hyperparameter combination is:\n")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    print(f"Best validation loss: {study.best_value:.4f}")

    best_params_file = os.path.join(args.save_dir, "best_hyperparams_ae.txt")
    with open(best_params_file, "w") as f:
        f.write("Best hyperparameter combination:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Best validation loss: {study.best_value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/content/jet_images_128.h5')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/JetTagging/anomaly_detection/optuna_logs')
    parser.add_argument('--tune_epochs', type=int, default=15, help="Epochs for each training")
    parser.add_argument('--warmup_epochs', type=int, default=4, help="Warmup epochs for pruning")
    parser.add_argument('--n_trials', type=int, default=20, help="Total number of trials")
    parser.add_argument('--img_size', type=int, default=128, help='Image size for resizing')
    parser.add_argument('--max_samples', type=int, default=100000, help="Maximum number of samples")
    args = parser.parse_args()
    
    main(args)