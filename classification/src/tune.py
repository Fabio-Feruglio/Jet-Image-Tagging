import argparse
import os
import optuna
import wandb
import torch
import torch.nn as nn
import numpy as np

# Assicurati che i path di importazione siano corretti nel tuo ambiente Colab
from dataset.dataloader import get_dataloaders
from model.resnet import ResNet50
from model.inception import InceptionV4

ARCHITECTURES = ["resnet", "inception"]

def build_actual_model(arch_name, num_classes=5):
    """Istanzia i veri modelli del progetto invece del modello fake linerare."""
    if arch_name == "resnet":
        return ResNet50(num_classes=num_classes)
    elif arch_name == "inception":
        return InceptionV4(num_classes=num_classes)
    else:
        raise ValueError(f"Architettura {arch_name} sconosciuta!")


def objective(trial, args, arch_name):
    """Esperimento Optuna per trovare i parametri ottimali del singolo backbone."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Iperparametri da ottimizzare
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    
    # Inizializzazione WandB per il trial corrente
    run = wandb.init(
        project="jet-tagging-tuning",
        group=f"optuna-{arch_name}",
        name=f"trial_{arch_name}_{trial.number}",
        config=trial.params,
        reinit=True
    )
    
    # Caricamento dati dinamico in base al batch_size proposto da Optuna
    train_loader, valid_loader, _ = get_dataloaders(
        data_filepath=args.data_path,  # Cambia da data_path a data_filepath
        img_size=args.img_size, 
        batch_size=batch_size, 
        max_samples=args.max_samples
    )
    
    model = build_actual_model(arch_name, num_classes=5).to(device)
    loss_fn = nn.CrossEntropyLoss()

    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)

    best_val_loss = float('inf')

    for epoch in range(args.tune_epochs):
        # Fase di Training
        model.train()
        train_losses = []
        correct_train = 0
        total_train = 0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            loss.backward()
            optimizer.step()
            
            train_losses.append(loss.item())
            correct_train += (outputs.argmax(dim=1) == batch_y).sum().item()
            total_train += batch_y.size(0)
            
        # Fase di Validazione
        model.eval()
        val_losses = []
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for batch_x, batch_y in valid_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
                outputs = model(batch_x)
                loss = loss_fn(outputs, batch_y)
                
                val_losses.append(loss.item())
                correct_val += (outputs.argmax(dim=1) == batch_y).sum().item()
                total_val += batch_y.size(0)
                
        epoch_train_loss = np.mean(train_losses)
        epoch_train_acc = correct_train / total_train
        epoch_val_loss = np.mean(val_losses)
        epoch_val_acc = correct_val / total_val
        
        # Log corretto su WandB
        wandb.log({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
        })
        
        # Pruning di Optuna se l'esperimento promette male
        trial.report(epoch_val_loss, epoch)
        if trial.should_prune():
            wandb.finish() 
            raise optuna.exceptions.TrialPruned()

        best_val_loss = min(best_val_loss, epoch_val_loss)

    wandb.finish()
    return best_val_loss


def run_epoch_final(model, loader, loss_fn, device, optimizer=None):
    """Funzione di utility per eseguire un'epoca completa (train o eval)."""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    losses = []
    correct = 0
    total = 0
    
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device).long()
            
            if is_train:
                optimizer.zero_grad()
                
            outputs = model(batch_x)
            loss = loss_fn(outputs, batch_y)
            
            if is_train:
                loss.backward()
                optimizer.step()
                
            losses.append(loss.item())
            correct += (outputs.argmax(dim=1) == batch_y).sum().item()
            total += batch_y.size(0)
            
    return np.mean(losses), correct / total


def train_final_model(arch_name, best_params, args):
    """Allena il modello finale con i parametri migliori trovati da Optuna."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
    train_loader, valid_loader, _ = get_dataloaders(
        data_path=args.data_path, 
        img_size=args.img_size, 
        batch_size=best_params["batch_size"],
        max_samples=args.max_samples
    )
 
    model = build_actual_model(arch_name, num_classes=5).to(device)
    loss_fn = nn.CrossEntropyLoss()
 
    if best_params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"]
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(), lr=best_params["lr"], momentum=0.9, weight_decay=best_params["weight_decay"]
        )
 
    wandb.init(
        project="jet-tagging-tuning",
        group="final-training",
        name=f"final-{arch_name}",
        config=best_params,
        reinit=True,
    )
 
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, f"{arch_name}_best.pth")
    best_val_loss = float("inf")
 
    for epoch in range(args.final_epochs):
        train_loss, train_acc = run_epoch_final(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_acc = run_epoch_final(model, valid_loader, loss_fn, device, optimizer=None)
 
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
        })
 
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({'model_state_dict': model.state_dict()}, ckpt_path)
 
    wandb.finish()
    print(f"🎯 Salvato checkpoint ottimo per {arch_name} in {ckpt_path} (val_loss={best_val_loss:.4f})")
    return ckpt_path


def evaluate_ensemble(models, loader, device):
    """Mette insieme i modelli calcolando la media delle probabilità Softmax."""
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

    # Allena ResNet e Inception singolarmente usando i loro iperparametri ideali
    ckpt_paths = {}
    for arch_name, params in best_params.items():
        ckpt_paths[arch_name] = train_final_model(arch_name, params, args)    

    # Carica i modelli completi per la valutazione dell'Ensemble
    models = []
    for arch_name in ARCHITECTURES:
        model = build_actual_model(arch_name, num_classes=5).to(device)
        checkpoint = torch.load(ckpt_paths[arch_name], map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        models.append(model)

    _, valid_loader, _ = get_dataloaders(data_path=args.data_path, img_size=args.img_size, batch_size=64, max_samples=args.max_samples)
    ens_loss, ens_acc = evaluate_ensemble(models, valid_loader, device)
 
    print("\n=== 🏆 RISULTATI ENSEMBLE (Validation) ===")
    print(f"Loss dell'Ensemble:      {ens_loss:.4f}")
    print(f"Accuratezza dell'Ensemble: {ens_acc:.4f}")        


def tune_architecture(arch_name, args):
    print(f"\n=== Tuning {arch_name} ===")
    study = optuna.create_study(
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=3),
    )
    study.optimize(lambda trial: objective(trial, args, arch_name), n_trials=args.n_trials)
    return study.best_trial.params


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/JetTagging/data/jet_images_299.h5')
    parser.add_argument('--checkpoint_dir', type=str, default='/content/drive/MyDrive/JetTagging/checkpoints')
    parser.add_argument('--tune_epochs', type=int, default=5, help="Epoche brevi per il tuning")
    parser.add_argument('--final_epochs', type=int, default=20, help="Epoche per l'addestramento finale serio")
    parser.add_argument('--n_trials', type=int, default=10, help="Numero di tentativi di tuning")
    parser.add_argument('--img_size', type=int, default=299)
    parser.add_argument('--max_samples', type=int, default=20000, help="Limitiamo i sample per non fondere Colab")
    
    args = parser.parse_args()
    main(args)
