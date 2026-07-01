import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score, confusion_matrix

from dataset import get_dataloaders
from model import InceptionV4

def evaluate_network(dataloader, model, loss_fn, device, data_split, save_dir):
    model.eval() # Use the evaluation mode for the network!
    
    with torch.no_grad(): # Remove gradient computation
        predictions = [] # Network output
        true = [] # True labels

        print(f"\n--- Valutazione sul set di {data_split.upper()} ---")
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            predictions.append(y_pred)
            true.append(batch_y)

        # Concatenate the batches in a single tensor
        predictions = torch.cat(predictions, dim=0)
        true = torch.cat(true, dim=0)

        # Compute the loss
        loss = loss_fn(predictions, true).item()

        # Convert to scikit-learn for metric evaluation
        probs = torch.sigmoid(predictions).detach().cpu().numpy().ravel()
        true_labels = true.detach().cpu().numpy().ravel()

        # Compute false-pos, true-pos and ROC curve thresholds
        fpr, tpr, thresholds = roc_curve(true_labels, probs)
        # AUC score (integral of the ROC curve)
        auc = roc_auc_score(true_labels, probs)

        # Convert probs to predicted labels, evaluate useful metrics for binary classification
        pred_labels = np.round(probs)
        precision, recall, fscore, _ = precision_recall_fscore_support(true_labels, pred_labels, average='binary')
        accuracy = accuracy_score(true_labels, pred_labels)

        print(f"\nRisultati {data_split}:")
        print(f"Loss:      {loss:.4f}")
        print(f"Accuracy:  {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall:    {recall:.4f}")
        print(f"F1 Score:  {fscore:.4f}")
        print(f"ROC AUC:   {auc:.4f}")
        print("\nConfusion Matrix:")
        print(confusion_matrix(true_labels, pred_labels))

        # ROC curve plot
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, lw=2, label=f'ROC curve (AUC = {auc:.2f})')
        plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color='gray')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        
        # Invece di plt.show(), salviamo il file! Perfetto per ambienti come Colab.
        plot_path = os.path.join(save_dir, f'roc_curve_{data_split}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"ROC plot saved in: {plot_path}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataloaders
    train_loader, valid_loader, test_loader = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)
    
    # Initialize the model and load weights
    model = InceptionV4().to(device)
    
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    loss_fn = nn.BCEWithLogitsLoss()
    
    # Evaluate
    evaluate_network(valid_loader, model, loss_fn, device, data_split="validation", save_dir=args.save_dir)
    evaluate_network(test_loader, model, loss_fn, device, data_split="test", save_dir=args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of ResNet50 model")
    parser.add_argument('--model_path', type=str, required=True, help="Model weights path")
    parser.add_argument('--data_dir', type=str, default='./data_lab04')
    parser.add_argument('--save_dir', type=str, default='./results', help="Directory for plots and results")
    parser.add_argument('--batch_size', type=int, default=64)
    
    args = parser.parse_args()
    main(args)