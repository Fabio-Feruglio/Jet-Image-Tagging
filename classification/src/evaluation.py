import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import accuracy_score, roc_curve, confusion_matrix, classification_report, auc
from sklearn.preprocessing import label_binarize

from dataset.dataloader import get_dataloaders
from model.resnet import ResNet50
from model.inception import InceptionV4
from model.ensemble import EnsembleModel

def evaluate_network(dataloader, model, loss_fn, device, data_split, save_dir, model_name, num_classes=5):
    model.eval() 
    
    with torch.no_grad(): # Remove gradient computation
        predictions = [] # Network output
        true = [] # True labels

        print(f"\n--- Eval on test set: {data_split.upper()} ---")
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            y_pred = model(batch_x)

            predictions.append(y_pred.cpu())
            true.append(batch_y.cpu())

        # Concatenate the batches in a single tensor
        predictions = torch.cat(predictions, dim=0)
        true = torch.cat(true, dim=0)

        # Compute the loss
        loss = loss_fn(predictions, true).item()

        # Convert to scikit-learn for metric evaluation
        probs = torch.softmax(predictions, dim=1).numpy()
        true_labels = true.numpy()
        pred_labels = np.argmax(probs, axis=1)

        # Test accuracy 
        accuracy = accuracy_score(true_labels, pred_labels)
        print(f"\nResults {data_split.upper()}:")
        print(f"Loss:     {loss:.4f}")
        print(f"Accuracy: {accuracy:.4f}")
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(true_labels, pred_labels))

        # Confusion matrix
        cm = confusion_matrix(true_labels, pred_labels)
        class_labels = [str(i) for i in range(num_classes)]
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot = True, fmt = 'd', cmap = 'Blues', 
                    xticklabels = class_labels, yticklabels = class_labels)
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        plt.title(f'Confusion Matrix - {data_split.capitalize()}')
        
        cm_path = os.path.join(save_dir, f'confusion_matrix_{data_split}_{model_name}.png')
        plt.savefig(cm_path, bbox_inches='tight')
        plt.close()
        print(f"Confusion Matrix saved in: {cm_path}")

        # Roc curve and AUC
        # One-hot encode the true labels for ROC computation
        true_labels_bin = np.asarray(label_binarize(true_labels, classes=list(range(num_classes)), sparse_output=False))
        
        plt.figure(figsize=(10, 8))
        
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(true_labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'Class {i} (AUC = {roc_auc:.3f})')

        plt.plot([0, 1], [0, 1], lw=2, linestyle='--', color='gray')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Multiclass ROC Curves - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(save_dir, f'roc_curve_{data_split}_{model_name}.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        print(f"ROC plot saved in: {roc_path}")

    return loss, accuracy

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataloaders
    _, valid_loader, test_loader = get_dataloaders(data_filepath = args.data_path, 
                                                   img_size = args.img_size, batch_size = args.batch_size, 
                                                   num_workers = min(4, os.cpu_count() or 1))
    
    # Initialize the model and load weights
    if args.model == 'resnet':
        model = ResNet50().to(device)
    elif args.model == 'inception':
        model = InceptionV4().to(device)
    elif args.model == 'ensemble':
        model = EnsembleModel(num_classes = 5, 
                              resnet_path = args.resnet_weights, 
                              inception_path = args.inception_weights, 
                              device = str(device)).to(device)
    else:
        raise ValueError("Invalid model type. Choose from 'resnet', 'inception', or 'ensemble'.")
    
    print(f"Loading model from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    loss_fn = nn.CrossEntropyLoss()
    
    # Evaluate
    evaluate_network(valid_loader, model, loss_fn, device, data_split="validation", save_dir=args.save_dir, model_name=args.model)
    evaluate_network(test_loader, model, loss_fn, device, data_split="test", save_dir=args.save_dir, model_name=args.model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of ResNet50 model")
    parser.add_argument('--model', type=str, default='resnet', choices=['resnet', 'inception', 'ensemble'], help="Choose the model to train: 'resnet', 'inception', or 'ensemble'")
    parser.add_argument('--model_path', type=str, required=True, help="Model weights path")
    parser.add_argument('--data_path', type=str, default='./data_lab04/jet_images_299.h5', help="Path to the dataset")
    parser.add_argument('--save_dir', type=str, default='./results', help="Directory for plots and results")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=299, help='Image size for resizing')

    args = parser.parse_args()
    main(args)