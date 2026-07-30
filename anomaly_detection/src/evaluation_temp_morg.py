import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
import pandas as pd
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset.dataloader import get_dataloaders 
from model.autoencoder import Encoder, Decoder

def evaluate_anomaly_detection(dataloader, encoder, decoder, device, save_dir, model_name, data_split):
    encoder.eval()
    decoder.eval()

    mse_loss_fn = nn.MSELoss(reduction='none') 
    
    anomaly_scores = []
    true_labels = []
    latent_vectors = []

    print(f"\n--- Eval on set: {data_split.upper()} ---")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            
            # Forward pass
            encoded = encoder(batch_x)
            reconstructed = decoder(encoded)

            # Anomaly score
            loss_per_pixel = mse_loss_fn(reconstructed, batch_x)
            loss_per_image = loss_per_pixel.view(loss_per_pixel.size(0), -1).mean(dim=1)

            # Save scores and true labels
            anomaly_scores.extend(loss_per_image.cpu().numpy())
            true_labels.extend(batch_y.numpy())
            
            # Save latent representations (appiattite per PCA/t-SNE)
            latent_vectors.extend(encoded.view(encoded.size(0), -1).cpu().numpy())

    anomaly_scores = np.array(anomaly_scores)
    true_labels = np.array(true_labels)
    latent_vectors = np.array(latent_vectors)
    
    mean_loss = np.mean(anomaly_scores)
    print(f"\nResults {data_split.upper()}:")
    print(f"Mean Reconstruction Loss: {mean_loss:.6f}")

    # ---------------------------------------------------------
    # PLOT 1: Anomaly Score Distribution
    # ---------------------------------------------------------
    plt.figure(figsize=(8, 8))
    
    sns.histplot(anomaly_scores[true_labels == 0], color='blue', label='Bg (q/g)', 
                 kde=True, stat='density', alpha=0.5, bins=70, binrange=(-0.2, 1.7))
    
    if np.sum(true_labels == 1) > 0:
        sns.histplot(anomaly_scores[true_labels == 1], color='red', label='Anomalies (t/W/Z)', 
                     kde=True, stat='density', alpha=0.5, bins=70, binrange=(-0.2, 1.7))
    plt.xlim(-0.2, 1.7)
    plt.ylim(0, None)
    plt.xlabel('Reconstruction Error (Anomaly Score)',fontsize=25, labelpad=10)
    plt.ylabel('Density',fontsize=25, labelpad=10)
    plt.title(f'Anomaly Score Distribution - {data_split.capitalize()}',fontsize=28, pad=15)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    dist_path = os.path.join(save_dir, f'loss_dist_{data_split}_{model_name}.pdf')
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved in: {dist_path}")

    # ---------------------------------------------------------
    # PLOT 2: ROC Curve 
    # ---------------------------------------------------------
    roc_auc = None
    if np.sum(true_labels == 1) > 0:
        fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        
        # Standard ROC Curve ---
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.xlabel('False Positive Rate', fontsize=25, labelpad=10)
        plt.ylabel('True Positive Rate', fontsize=25, labelpad=10)
        plt.title(f'Standard ROC Curve - {data_split.capitalize()}', fontsize=28, pad=15)
        plt.legend(loc="lower right", fontsize=20)
        
        roc_path = os.path.join(save_dir, f'roc_curve_{data_split}_{model_name}.pdf')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()

        # Background Rejection (e_S vs 1/e_B) ---
        plt.figure(figsize=(8, 8))
        
        valid_idx = fpr > 0
        fpr_valid = fpr[valid_idx]
        tpr_valid = tpr[valid_idx]
        
        # Rejection = 1 / FPR
        rejection = 1.0 / fpr_valid
        
        plt.plot(tpr_valid, rejection, color='purple', lw=2, label=f'Autoencoder Rejection')
        plt.yscale('log') 
        plt.xlim((0.0, 1.0))
        
        plt.xlabel('Signal Efficiency ($\epsilon_S$)', fontsize=25, labelpad=10)
        plt.ylabel('Background Rejection ($1/\epsilon_B$)', fontsize=25, labelpad=10)
        plt.title(f'Background Rejection Curve - {data_split.capitalize()}', fontsize=28, pad=15)
        plt.grid(True, which="both", ls="--", alpha=0.5)
        plt.legend(loc="upper right", fontsize=20)
        
        rej_path = os.path.join(save_dir, f'rejection_curve_{data_split}_{model_name}.pdf')
        plt.savefig(rej_path, bbox_inches='tight')
        plt.close()
        
        print(f"ROC and Rejection plots saved in: {save_dir}")
    else:
        print(f"Skipping ROC and Rejection curves for {data_split.upper()} (No anomalies present).")

    # ---------------------------------------------------------
    # PLOT 3: Latent Space Projections (PCA & t-SNE)
    # ---------------------------------------------------------
    print(f"Computing PCA and t-SNE for latent space projection ({data_split.upper()})")
    
    label_names = {0: 'Bg (q/g)', 1: 'Anomalies (t/W/Z)'}
    mapped_labels = [label_names[l] for l in true_labels]

    # PCA 
    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latent_pca[:, 0], y=latent_pca[:, 1], hue=mapped_labels, 
                    palette={'Bg (q/g)': 'blue', 'Anomalies (t/W/Z)': 'red'}, 
                    alpha=0.6)
    plt.title(f'PCA Latent Space Projection - {data_split.capitalize()}', fontsize=28, pad=15)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlabel('Principal Component 1', fontsize=25, labelpad=10)
    plt.ylabel('Principal Component 2', fontsize=25, labelpad=10)
    plt.legend(fontsize=20)
    
    pca_path = os.path.join(save_dir, f'pca_latent_{data_split}_{model_name}.pdf')
    plt.savefig(pca_path, bbox_inches='tight')
    plt.close()
    print(f"PCA plot saved in: {pca_path}")

    # t-SNE 
    # Note: t-SNE can be slow for large datasets; consider subsampling if needed
    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latent_tsne[:, 0], y=latent_tsne[:, 1], hue=mapped_labels, 
                    palette={'Bg (q/g)': 'blue', 'Anomalies (t/W/Z)': 'red'}, 
                    alpha=0.6)
    plt.title(f't-SNE Latent Space Projection - {data_split.capitalize()}', fontsize=28, pad=15)
    plt.xlabel('t-SNE Dimension 1', fontsize=25, labelpad=10)
    plt.ylabel('t-SNE Dimension 2', fontsize=25, labelpad=10)
    plt.legend(fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    tsne_path = os.path.join(save_dir, f'tsne_latent_{data_split}_{model_name}.pdf')
    plt.savefig(tsne_path, bbox_inches='tight')
    plt.close()
    print(f"t-SNE plot saved in: {tsne_path}")

    return mean_loss, roc_auc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataloaders
    _, valid_loader, test_loader = get_dataloaders(
        data_filepath = args.data_path, 
        bg_classes = args.bg_classes,
        img_size = args.img_size, 
        batch_size = args.batch_size, 
        num_workers = min(4, os.cpu_count() or 1),
        max_samples = args.max_samples
    )
    
    # Initialize the Autoencoder model
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    
    print(f"Loading model weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
        decoder.load_state_dict(checkpoint)

    # Evaluate on validation
    evaluate_anomaly_detection(valid_loader, encoder, decoder, device, data_split="validation", 
                               save_dir=args.save_dir, model_name="autoencoder")
    
    # Evaluate on test
    evaluate_anomaly_detection(test_loader, encoder, decoder, device, data_split="test", 
                               save_dir=args.save_dir, model_name="autoencoder")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of Autoencoder for Anomaly Detection")
    parser.add_argument('--model_path', type=str, required=True, help="Autoencoder weights path")
    parser.add_argument('--data_path', type=str, default='./dataset.h5', help="Path to the dataset")
    parser.add_argument('--save_dir', type=str, default='./results_ad', help="Directory for plots and results")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1], help='Classes to consider as background (e.g. 0 1)')
    parser.add_argument('--latent_space_dim', type=int, default=128, help='Dimension of the latent space')
    args = parser.parse_args()
    main(args)