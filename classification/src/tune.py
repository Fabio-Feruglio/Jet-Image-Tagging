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
    
    # HyperParameters to be optimized
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    print(f"\nTrial {trial.number}: lr={lr}, batch_size={batch_size}, weight_decay={weight_decay}")

    
    # Initialize WandB: we can use this instead of TensorBoard, for better sharing
    run = wandb.init(
        project = "jet-tagging-tuning",   # Project name
        group = f"optuna-{args.model}",   # group experiment runs
        config = trial.params,            # parameters of each experiment
        reinit = True                     # reinitialize network each time
    )
    
    # Load the data and build the model
    train_dataloader, valid_dataloader, _ = get_dataloaders(data_filepath = args.data_path, 
                                                            img_size = args.img_size, 
                                                            batch_size = batch_size, 
                                                            num_workers = min(4, os.cpu_count() or 1),
                                                            max_samples = args.max_samples)
    
    model = build_model(trial, args.model).to(device)
    loss_fn = nn.CrossEntropyLoss()

 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay = weight_decay)
    best_val_loss = float('inf')

    # Training cycle (fixed number of epochs for each experiment: max 10)
    for epoch in range(args.tune_epochs):
        model.train()
        train_losses = []
        correct_train = 0
        total_train = 0
        for batch_x, batch_y in train_dataloader:
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
        print(f"EPOCH {epoch+1}/{args.tune_epochs} | Train Loss: {np.mean(train_losses):.4f} | Train Acc: {correct_train/total_train:.4f}")

        model.eval()
        val_losses = []
        correct_val = 0
        total_val = 0
        with torch.no_grad():
            for batch_x, batch_y in valid_dataloader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_losses.append(loss_fn(outputs, batch_y).item())
                _, predicted = torch.max(outputs.data, 1)
                correct_val += (predicted == batch_y).sum().item()
                total_val += batch_y.size(0)
        print(f"EPOCH {epoch+1}/{args.tune_epochs} | Val Loss: {np.mean(val_losses):.4f} | Val Acc: {correct_val/total_val:.4f}")

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

    # Save the best hyperparameters to a file
    best_params_file = os.path.join(args.data_path, f"best_hyperparams_{args.model}.txt")
    with open(best_params_file, "w") as f:
        f.write("Best hyperparameter combination:\n")
        for key, value in study.best_trial.params.items():
            f.write(f"{key}: {value}\n")
        f.write(f"Best validation loss: {study.best_value:.4f}\n")

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