import argparse
import os
from tqdm import tqdm
import optuna
import wandb
import torch
import torch.nn as nn
import numpy as np

# Assicurati che l'import del DataLoader e del VAE corrispondano ai file reali
from dataset.dataloader import get_dataloaders
from VAE import VAE_Ensemble_Light 

def set_backbone_trainable(model, trainable: bool):
    """
    Activate or deactivate the training of the backbone models (ResNet e Inception).
    """
    for param in model.encoder.Vencoder.resnet.parameters():
        param.requires_grad = trainable
    for param in model.encoder.Vencoder.inception.parameters():
        param.requires_grad = trainable

def vae_loss_fn(recon_x, x, mu, log_var, sigma=1.0):
    """
    Loss per il VAE composta da errore di ricostruzione + divergenza KL
    """
    # Mean Squared Error Loss per la ricostruzione
    # 'sum' somma gli errori su tutti i pixel per ogni immagine del batch
    recon_loss = (nn.MSELoss(reduction='sum')(recon_x, x))/(2*sigma**2)
    
    # Kullback-Leibler Divergence
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    
    return recon_loss + kld_loss

def objective(trial, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Hyperparameters
    encoded_space_dim = trial.suggest_categorical("encoded_space_dim", [128, 256, 512])
    lr_mlp = trial.suggest_categorical("lr_mlp", [1e-4, 1e-3, 1e-2])                            # LR per FC ed encoder/decoder
    lr_backbone = trial.suggest_categorical("lr_backbone", [1e-6, 1e-5, 1e-4])                  # LR per i backbone convoluzionali
    weight_decay = trial.suggest_categorical("weight_decay", [1e-6, 1e-5, 1e-4, 1e-3, 1e-2])    # Weight decay
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])                         # Batch size
    frozen_epochs = trial.suggest_int("frozen_epochs", 0, 5) 
    sigma = trial.suggest_float("sigma", 0.1, 10.0)                                    # Epochs per il freeze
    
    print(f"\nTRIAL {trial.number}\n{'='*40}")
    print("Hyperparameters:")
    print(f"Encoded Space Dim: {encoded_space_dim}")
    print(f"LR MLP: {lr_mlp:.6f}, LR Backbone: {lr_backbone:.6f}")
    print(f"Weight Decay: {weight_decay:.6f}, Batch size: {batch_size}, Frozen Epochs: {frozen_epochs}")
    
    wandb.init(
        project = "jet-tagging-tuning",
        group = "optuna-vae-ensemble",
        name = f"trial_{trial.number}",
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
    
    model = VAE_Ensemble_Light(
        encoded_space_dim = encoded_space_dim,
        im_size = args.img_size
    ).to(device)
    
    # Optimizer separato per sezioni del VAE
    optimizer = torch.optim.Adam([
        {'params': model.encoder.Vencoder.resnet.parameters(), 'lr': lr_backbone},
        {'params': model.encoder.Vencoder.inception.parameters(), 'lr': lr_backbone},
        {'params': model.encoder.Vencoder.fc.parameters(), 'lr': lr_mlp},
        {'params': model.encoder.fc_mu.parameters(), 'lr': lr_mlp},
        {'params': model.encoder.fc_var.parameters(), 'lr': lr_mlp},
        {'params': model.decoder.parameters(), 'lr': lr_mlp}
    ], weight_decay = weight_decay)

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')

    for epoch in range(args.tune_epochs):
        
        if epoch < frozen_epochs:
            set_backbone_trainable(model, False)
            model.encoder.Vencoder.resnet.eval()
            model.encoder.Vencoder.inception.eval()
            
            # Layer custom mantengono la modalità train
            model.encoder.Vencoder.fc.train()
            model.encoder.fc_mu.train()
            model.encoder.fc_var.train()
            model.decoder.train()
            status = "Frozen Backbones"
        else:
            set_backbone_trainable(model, True)
            model.train()
            status = "Fine-Tuning"

        train_losses = []
        print(f"\nEPOCH {epoch+1}/{args.tune_epochs} [{status}]")

        ### TRAINING LOOP
        for batch_x, _ in tqdm(train_dataloader, desc = "Training"):
            # Il VAE ignora batch_y, usa le immagini come input e come target
            batch_x = batch_x.to(device)
            
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                
                # Forward: PyTorch rispetta i gradienti bloccati, non serve una ramificazione esplicita
                x_recon, mu, log_var = model(batch_x)
                loss = vae_loss_fn(x_recon, batch_x, mu, log_var)
            
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
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, _ in tqdm(valid_dataloader, desc = "Validation"):
                batch_x = batch_x.to(device)
                
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    x_recon, mu, log_var = model(batch_x)
                    loss = vae_loss_fn(x_recon, batch_x, mu, log_var)
                
                val_losses.append(loss.item())
                
        print(f"Val Loss: {np.mean(val_losses):.4f}")

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss,
            "backbone_status": 0 if epoch < frozen_epochs else 1 # 0 = Frozen, 1 = Unfrozen
        })
        
        # Optuna Pruning
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            wandb.finish() 
            print(f"Trial pruned at epoch {epoch+1} con val loss {epoch_val_loss:.4f}")
            raise optuna.exceptions.TrialPruned()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
    
    print(f"Trial {trial.number} completato con best val loss: {best_val_loss:.4f}")
    wandb.finish()
    
    return float(best_val_loss)

def main(args):
    print("Optimization of VAE hyperparameters...")

    os.makedirs(args.save_dir, exist_ok=True)
    db_path = os.path.join(args.save_dir, "optuna_study_vae.db")
    storage_name = f"sqlite:///{db_path}"
    study_name = "jet_tagging_study_vae"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.warmup_epochs)
    )
    
    print(f"Already completed trials: {len(study.trials)}")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print("The best hyperparameter combination is:\n")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    print(f"Best validation loss: {study.best_value:.4f}")

    best_params_file = os.path.join(args.save_dir, "best_hyperparams_vae.txt")
    with open(best_params_file, "w") as f:
        f.write("Best hyperparameter combination:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Best validation loss: {study.best_value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/JetTagging/data/jet_images_299.h5')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/JetTagging/optuna_logs')
    
    parser.add_argument('--tune_epochs', type=int, default=15, help="Epochs for each training")
    parser.add_argument('--warmup_epochs', type=int, default=6, help="Warmup epochs for pruning")
    parser.add_argument('--n_trials', type=int, default=20, help="Total number of trials")
    parser.add_argument('--img_size', type=int, default=299, help='Image size for resizing')
    parser.add_argument('--max_samples', type=int, default=20000, help="Maximum number of samples")
    args = parser.parse_args()
    
    main(args)