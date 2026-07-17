import argparse
import os
from tqdm import tqdm
import optuna
import wandb
import torch
import torch.nn as nn
import numpy as np

from dataset.dataloader import get_dataloaders
from model.resnet import ResNet50
from model.inceptionv3 import InceptionV3
from model.ensemble import EnsembleModel

# Example file for tuning with optuna and viewing training loss / validation loss with Wandb

def build_model(model_name):
    """
    Use optuna 'trial' object to define a dynamic model
    """
    if model_name == "resnet":
        model = ResNet50(num_classes = 5)  
        return model
    elif model_name == "inception":
        model = InceptionV3(num_classes = 5)  
        return model
    elif model_name == "ensemble":
        model = EnsembleModel(num_classes = 5)
        return model
    else:
        raise ValueError(f"Model {model_name} not supported.")

def objective(trial, args):
    """
    Optuna experiment function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # HyperParameters to be optimized
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log = True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log = True)

    print(f"\nTRIAL {trial.number}\n{'='*40}")
    print(f"Learning rate: {lr}, Batch size: {batch_size}, Weight decay: {weight_decay}")
    
    # Initialize WandB: we can use this instead of TensorBoard, for better sharing
    wandb.init(
        project = "jet-tagging-tuning",   # Project name
        group = f"optuna-{args.model}",   # group experiment runs
        name = f"trial_{trial.number}",   # name of the run
        config = trial.params,            # parameters of each experiment
        reinit = True                   # reinitialize network each time
    )
    
    # Load the data and build the model
    train_dataloader, valid_dataloader, _ = get_dataloaders(data_filepath = args.data_path, 
                                                            img_size = args.img_size, 
                                                            batch_size = batch_size, 
                                                            num_workers = min(4, os.cpu_count() or 1),
                                                            max_samples = args.max_samples)
    
    model = build_model(args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()

 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    best_val_loss = float('inf')

    # Training cycle (fixed number of epochs for each experiment: max 10)
    for epoch in range(args.tune_epochs):
        model.train()
        train_losses = []
        correct_train = 0
        total_train = 0
        print (f"\nEPOCH {epoch+1}/{args.tune_epochs}")
        for batch_x, batch_y in tqdm(train_dataloader, desc="Training"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            _, predicted = torch.max(outputs.data, 1)
            correct_train += (predicted == batch_y).sum().item()
            total_train += batch_y.size(0)
        print(f"Train Loss: {np.mean(train_losses):.4f} | Train Acc: {correct_train/total_train:.4f}")

        model.eval()
        val_losses = []
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in tqdm(valid_dataloader, desc="Validation"):
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_losses.append(loss_fn(outputs, batch_y).item())
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)
        print(f"Val Loss: {np.mean(val_losses):.4f} | Val Acc: {correct_val/total_val:.4f}")

        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        epoch_train_acc = correct_train / total_train
        epoch_val_acc = correct_val / total_val
        
        # WandB logger
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc
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
    print("Optimization of hyperparameters...")

    # Create save directory and database path for Optuna study
    os.makedirs(args.save_dir, exist_ok = True)
    db_path = os.path.join(args.save_dir, f"optuna_study_{args.model}.db")
    storage_name = f"sqlite:///{db_path}"
    study_name = f"jet_tagging_study_{args.model}"
    
    # Create optuna study (if it already exists, load it)
    study = optuna.create_study(
        study_name = study_name,
        storage = storage_name,
        load_if_exists = True,
        direction = "minimize", 
        pruner = optuna.pruners.MedianPruner(n_warmup_steps = args.warmup_epochs) # at least 3 epochs before eventually stopping
    )
    
    print(f"Already completed trials: {len(study.trials)}")

    
    # Objective function: n_trials tells how many experiments we want
    study.optimize(lambda trial: objective(trial, args), n_trials = args.n_trials)
    
    print("\n")
    print("The best hyperparameter combination is:\n")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}\n")
        
    print(f"Best validation loss: {study.best_value:.4f}")

    # Save the best hyperparameters to a file
    best_params_file = os.path.join(args.save_dir, f"best_hyperparams_{args.model}.txt")
    with open(best_params_file, "w") as f:
        f.write("Best hyperparameter combination:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Best validation loss: {study.best_value:.4f}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/JetTagging/data/jet_images_299.h5', help='Path to the dataset')
    parser.add_argument('--save_dir', type=str, default='/content/drive/MyDrive/JetTagging/optuna_logs', help='Directory to save checkpoints andthe best hyperparameters')
    parser.add_argument('--tune_epochs', type=int, default=10, help="Epochs for each training (keep low: max 10-15)")
    parser.add_argument('--warmup_epochs', type=int, default=4, help="Warmup epochs for pruning")
    parser.add_argument('--n_trials', type=int, default=20, help="Total number of trials")
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'inception', 'ensemble'], help="Model name")
    parser.add_argument('--img_size', type=int, default=299, help='Image size for resizing')
    parser.add_argument('--max_samples', type=int, default=20000, help="Maximum number of samples to use for tuning")
    args = parser.parse_args()
    main(args)