import argparse
import os
from tqdm import tqdm
import optuna
import wandb
import torch
import torch.nn as nn
import numpy as np

# Assicurati che questi import puntino ai percorsi corretti nel tuo progetto
from dataset.dataloader import get_dataloaders
from model.ensemble import EnsembleModel

def set_backbone_trainable(model, trainable: bool):
    """
    Attiva o disattiva il calcolo dei gradienti per i backbone.
    """
    for param in model.resnet.parameters():
        param.requires_grad = trainable
    for param in model.inception.parameters():
        param.requires_grad = trainable

def objective(trial, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. HyperParameters specifici per l'Ensemble
    # LR per i layer finali (solitamente più alto perché partono da zero)
    lr_mlp = trial.suggest_float("lr_mlp", 1e-4, 1e-2, log=True)
    # LR per i backbone (solitamente più basso per non distruggere i pesi pre-addestrati)
    lr_backbone = trial.suggest_float("lr_backbone", 1e-6, 1e-4, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [ 32, 64, 128 ])
    
    # Epoche in cui i backbone rimangono "congelati" (es. da 0 a metà delle epoche totali)
    max_frozen_epochs = max(0, args.tune_epochs // 2)
    frozen_epochs = trial.suggest_int("frozen_epochs", 0, max_frozen_epochs)

    print(f"\nTRIAL {trial.number}\n{'='*40}")
    print(f"LR MLP: {lr_mlp:.6f}, LR Backbone: {lr_backbone:.6f}")
    print(f"Batch size: {batch_size}, Frozen Epochs: {frozen_epochs}")
    
    wandb.init(
        project="jet-tagging-tuning",
        group="optuna-ensemble",
        name=f"trial_{trial.number}",
        config=trial.params,
        reinit=True
    )
    
    train_dataloader, valid_dataloader, _ = get_dataloaders(
        data_filepath=args.data_path, 
        img_size=args.img_size, 
        batch_size=batch_size, 
        num_workers=min(4, os.cpu_count() or 1),
        max_samples=args.max_samples
    )
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    # Inizializza il modello caricando i pesi se forniti
    model = EnsembleModel(
        num_classes=5, 
        resnet_path=args.resnet_path, 
        inception_path=args.inception_path, 
        device=device,
        dropout=dropout
    ).to(device)
    
    loss_fn = nn.CrossEntropyLoss()

    # 2. Parameter Groups: assegniamo i learning rate separati
    optimizer = torch.optim.Adam([
        {'params': model.resnet.parameters(), 'lr': lr_backbone},
        {'params': model.inception.parameters(), 'lr': lr_backbone},
        {'params': model.fc.parameters(), 'lr': lr_mlp}
    ], weight_decay=weight_decay)

    # ---> AGGIUNGI LO SCALER PER LA MIXED PRECISION <---
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    best_val_loss = float('inf')

    for epoch in range(args.tune_epochs):
        
        if epoch < frozen_epochs:
            set_backbone_trainable(model, False)
            model.resnet.eval()
            model.inception.eval()
            model.fc.train()
            status = "FROZEN Backbones"
        else:
            set_backbone_trainable(model, True)
            model.train()
            status = "UNFROZEN Backbones (Fine-Tuning)"

        train_losses = []
        correct_train = 0
        total_train = 0
        print(f"\nEPOCH {epoch+1}/{args.tune_epochs} [{status}]")
        
        for batch_x, batch_y in tqdm(train_dataloader, desc="Training"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            
            # ---> USA L'AUTOCAST PER VELOCIZZARE LA GPU <---
            with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                
                # ---> VELOCIZZA I BACKBONE CONGELATI <---
                if epoch < frozen_epochs:
                    with torch.no_grad(): # Evita di calcolare il grafo se sono congelati
                        resnet_out = model.resnet(batch_x)
                        inception_out = model.inception(batch_x)
                        combined_out = torch.cat((resnet_out, inception_out), dim=1)
                    # Solo l'MLP finale richiede il grafo
                    outputs = model.fc(combined_out)
                else:
                    # Fine tuning completo
                    outputs = model(batch_x)
                
                loss = loss_fn(outputs, batch_y)
            
            # ---> BACKWARD PASS OTTIMIZZATO CON LO SCALER <---
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        print(f"Train Loss: {np.mean(train_losses):.4f} | Train Acc: {correct_train/total_train:.4f}")

        # ---> VALIDATION PHASE (Anche qui va usato autocast!) <---
        model.eval()
        val_losses = []
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(valid_dataloader, desc="Validation"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                with torch.amp.autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    outputs = model(batch_x)
                    loss = loss_fn(outputs, batch_y)
                
                val_losses.append(loss.item())
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)
                
        print(f"Val Loss: {np.mean(val_losses):.4f} | Val Acc: {correct_val/total_val:.4f}")

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": correct_train / total_train,
            "val_loss": epoch_val_loss,
            "val_acc": correct_val / total_val,
            "backbone_status": 0 if epoch < frozen_epochs else 1 # 0=Frozen, 1=Unfrozen
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
    print("Optimization of Ensemble hyperparameters...")

    os.makedirs(args.save_dir, exist_ok=True)
    db_path = os.path.join(args.save_dir, "optuna_study_ensemble.db")
    storage_name = f"sqlite:///{db_path}"
    study_name = "jet_tagging_study_ensemble"
    
    study = optuna.create_study(
        study_name=study_name,
        storage=storage_name,
        load_if_exists=True,
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=args.warmup_epochs)
    )
    
    print(f"Trials già completati: {len(study.trials)}")
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print("\nLa migliore combinazione di iperparametri è:\n")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    print(f"\nBest validation loss: {study.best_value:.4f}")

    best_params_file = os.path.join(args.save_dir, "best_hyperparams_ensemble.txt")
    with open(best_params_file, "w") as f:
        f.write("Best hyperparameter combination:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Best validation loss: {study.best_value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/JetTagging/data/jet_images_299.h5')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/JetTagging/optuna_logs')
    
    parser.add_argument('--resnet_path', type=str, default=None, help='Path ai pesi preaddestrati di ResNet')
    parser.add_argument('--inception_path', type=str, default=None, help='Path ai pesi preaddestrati di Inception')
    
    parser.add_argument('--tune_epochs', type=int, default=15, help="Epochs for each training")
    parser.add_argument('--warmup_epochs', type=int, default=4, help="Warmup epochs for pruning")
    parser.add_argument('--n_trials', type=int, default=20, help="Total number of trials")
    parser.add_argument('--img_size', type=int, default=299, help='Image size for resizing')
    parser.add_argument('--max_samples', type=int, default=20000, help="Maximum number of samples")
    args = parser.parse_args()
    
    main(args)