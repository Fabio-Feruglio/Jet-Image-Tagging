import argparse
import os
import optuna
import wandb
import torch
import torch.nn as nn
import numpy as np

from dataset.dataloader import get_dataloaders
from model.resnet import ResNet50
from model.inception import InceptionV4

# Example file for tuning with optuna and viewing training loss / validation loss with Wandb

def build_model_example(trial):
    """
    Use optuna 'trial' object to define a dynamic model
    NB: this is an example model: we could use directly the fixed ResNet class
        eventually modifying it to include some modifiable parameters (e.g. dropout rate)
    """
    # HyperPar 1: number of layers at the end of the ResNet
    n_layers = trial.suggest_int("n_layers", 1, 3)
    
    layers = []
    in_features = 10 
    
    for i in range(n_layers):
        # HyperPar2 2: Neurons for each layer
        out_features = int(trial.suggest_categorical(f"n_units_l{i}", [16, 32, 64, 128]))
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.ReLU())
        
        # HyperPar 3: Dropout probability
        dropout_rate = trial.suggest_float(f"dropout_l{i}", 0.1, 0.5)
        layers.append(nn.Dropout(dropout_rate))
        
        in_features = out_features
        
    layers.append(nn.Linear(in_features, 1)) # Final layer for binary classification
    return nn.Sequential(*layers)

def build_model(trial, model_name):
    """
    Use optuna 'trial' object to define a dynamic model
    """
    if model_name == "resnet":
        model = ResNet50(num_classes=5)  
        return model
    elif model_name == "inception":
        model = InceptionV4(num_classes=5)  
        return model
    else:
        raise ValueError(f"Model {model_name} not supported.")

def objective(trial, args):
    """
    Optuna experiment function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # HyperPar 1: learning rate (log scale)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # HyperPar 2: batch size
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    
    # Initialize WandB: we can use this instead of TensorBoard, for better sharing
    run = wandb.init(
        project="jet-tagging-classification", # Project name
        group="optuna-study-01",   # group experiment runs
        config=trial.params,       # parameters of each experiment
        reinit=True                # reinitialize network each time
    )
    
    # Load the data and build the model
    train_dataloader, valid_dataloader, _ = get_dataloaders(data_filepath = args.data_path, 
                                                            img_size = args.img_size, batch_size = batch_size, 
                                                            num_workers = min(4, os.cpu_count() or 1),
                                                            max_samples = args.max_samples)
    
    model = build_model(trial, args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()

    # HyperPar 3: optimizer choice
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    best_val_loss = float('inf')

    # Training cycle (fixed number of epochs for each experiment: max 10)
    for epoch in range(args.tune_epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_dataloader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1).float()
            
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in valid_dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1).float()
                val_losses.append(loss_fn(model(batch_x), batch_y).item())
                
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        
        # WandB logger
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "val_loss": epoch_val_loss
        })
        
        # Optuna Pruning
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            wandb.finish() 
            raise optuna.exceptions.TrialPruned()

        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss

    wandb.finish()
    
    return best_val_loss

def main(args):
    print("Optimization of hyperparameters...")
    
    # Create optuna study
    study = optuna.create_study(
        direction="minimize", 
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3) # at least 3 epochs before eventually stopping
    )
    
    # Objective function: n_trials tells how many experiments we want
    study.optimize(lambda trial: objective(trial, args), n_trials=args.n_trials)
    
    print("\n")
    print("The best hyperparameter combination is:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
        
    print(f"Best validation loss: {study.best_value:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data_lab04')
    parser.add_argument('--tune_epochs', type=int, default=10, help="Epochs for each training (keep low: max 10-15)")
    parser.add_argument('--n_trials', type=int, default=20, help="Total number of trials")
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'inception'], help="Model name")
    parser.add_argument('--img_size', type=int, default=299, help='Image size for resizing')
    parser.add_argument('--max_samples', type=int, default=20000, help="Maximum number of samples to use for tuning")
    args = parser.parse_args()
    main(args)