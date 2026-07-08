import argparse
import os
import optuna
import wandb
import torch
import torch.nn as nn
import numpy as np
import os

from dataloader import get_dataloaders

# Example file for tuning with optuna and viewing training loss / validation loss with Wandb

def build_model(trial):
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



def objective(trial, args, arch_name):
    """
    Optuna experiment function
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # HyperPar 1: learning rate (log scale)
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    # HyperPar 2: batch size
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    # HyperPar 6: optimizer choice
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    # Initialize WandB: we can use this instead of TensorBoard, for better sharing
    run = wandb.init(
        project="wifi-har-tuning", # Project name
        group="optuna-study-01",   # group experiment runs
        config=trial.params,       # parameters of each experiment
        reinit=True                # reinitialize network each time
    )
    
    # Load the data and build the model
    train_dataloader, valid_dataloader, _ = get_dataloaders(data_filepath = args.data_path, 
                                                            img_size = args.img_size, batch_size = batch_size, 
                                                            num_workers = min(4, os.cpu_count() or 1),
                                                            max_samples = args.max_samples)
    
    model = build_model(trial).to(device)
    loss_fn = nn.BCEWithLogitsLoss()

    # HyperPar 6: optimizer choice
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    best_val_loss = float('inf')

    # Training cycle (fixed number of epochs for each experiment: max 10)
    for epoch in range(args.tune_epochs):
        model.train()
        train_losses = []
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1).float()
            
            optimizer.zero_grad()
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).unsqueeze(1).float()
                val_losses.append(loss_fn(model(batch_x), batch_y).item())
                
        epoch_train_loss = np.mean(train_losses)
        epoch_val_loss = np.mean(val_losses)
        
        # WandB logger
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
        
        # Optuna Pruning
        trial.report(val_loss, epoch)
        if trial.should_prune():
            wandb.finish() 
            raise optuna.exceptions.TrialPruned()

        best_val_loss = min(best_val_loss, val_loss)

    wandb.finish()
    
    return best_val_loss

#tuniamo per InceptionV4 e Resnet50 separatamente, così da avere due studi indipendenti
def tune_architecture(arch_name, args):
    """Runs an independent Optuna study for one backbone."""
    print(f"\n=== Tuning {arch_name} ===")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(lambda trial: objective(trial, args, arch_name), n_trials=args.n_trials)
 
    print(f"Best hyperparameters for {arch_name}:")
    for key, value in study.best_trial.params.items():
        print(f"    {key}: {value}")
    print(f"Best val loss ({arch_name}): {study.best_value:.4f}")
 
    return study.best_trial.params


# retrain with best hyperparameters for the full number of epochs and save the best model

def train_final_model(arch_name, best_params, args):
    """Retrains a model with the best hyperparameters for the full number of
    epochs and checkpoints the best-val-loss weights, for later ensembling."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    train_loader, valid_loader, _ = get_dataloaders(
        data_dir=args.data_dir, batch_size=best_params["batch_size"]
    )
 
    model = build_model(arch_name, args.num_classes).to(device)
    loss_fn = nn.CrossEntropyLoss()
 
    if best_params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=best_params["lr"], momentum=0.9,
            weight_decay=best_params["weight_decay"],
        )
 
    wandb.init(
        project="jet-tagging-tuning",
        group="final-training",
        name=f"final-{arch_name}",
        config=best_params,
        reinit=True,
    )
 
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{arch_name}_best.pt")
    best_val_loss = float("inf")
 
    for epoch in range(args.final_epochs):
        train_loss, train_acc = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch(model, valid_loader, loss_fn, device, optimizer=None)
 
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), ckpt_path)
 
    wandb.finish()
    print(f"Saved best {arch_name} checkpoint to {ckpt_path} (val_loss={best_val_loss:.4f})")
    return ckpt_path

# evaluate the ensemble of models on the test set, reporting loss and accuracy

def evaluate_ensemble(models, loader, device):
    """Averages softmax probabilities across models; reports loss/accuracy."""
    for m in models:
        m.eval()
 
    nll = nn.NLLLoss()
    losses, correct, total = [], 0, 0
 
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device).long()
 
            probs_per_model = [torch.softmax(m(batch_x), dim=1) for m in models]
            avg_probs = torch.stack(probs_per_model, dim=0).mean(dim=0)
 
            loss = nll(torch.log(avg_probs.clamp_min(1e-12)), batch_y)
            losses.append(loss.item())
 
            correct += (avg_probs.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)
 
    return float(np.mean(losses)), correct / total


def main(args):
    

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    best_params = {}
    for arch_name in ARCHITECTURES:
        best_params[arch_name] = tune_architecture(arch_name, args)

    # Retrain each backbone with its own best hyperparameters (full run)
    ckpt_paths = {}
    for arch_name, params in best_params.items():
        ckpt_paths[arch_name] = train_final_model(arch_name, params, args)    

    models = []
    for arch_name in ARCHITECTURES:
        model = build_model(arch_name, args.num_classes).to(device)
        model.load_state_dict(torch.load(ckpt_paths[arch_name], map_location=device))
        models.append(model)

    _, valid_loader, _ = get_dataloaders(data_dir=args.data_dir, batch_size=64)
    ens_loss, ens_acc = evaluate_ensemble(models, valid_loader, device)
 
    print("\n=== Ensemble results (validation set) ===")
    print(f"Ensemble loss:     {ens_loss:.4f}")
    print(f"Ensemble accuracy: {ens_acc:.4f}")        
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./data_lab04')
    parser.add_argument('--tune_epochs', type=int, default=10, help="Epochs for each training (keep low: max 10-15)")
    parser.add_argument('--n_trials', type=int, default=20, help="Total number of trials")
    
    args = parser.parse_args()
    main(args)