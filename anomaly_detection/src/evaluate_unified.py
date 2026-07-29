import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def evaluate_unified(dataloader, encoder, decoder, device, save_dir, model_name, data_split, args):
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
            
            if args.model == 'vae':
                encoded, mu, log_var = encoder(batch_x)
                reconstructed = decoder(encoded)
                
                sigma = 1.0
                loss_per_pixel = mse_loss_fn(reconstructed, batch_x)
                recon_loss_per_image = loss_per_pixel.view(loss_per_pixel.size(0), -1).mean(dim=1) / (sigma**2)
                kl_div_per_image = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
                num_pixels = batch_x.shape[1] * batch_x.shape[2] * batch_x.shape[3]
                kl_scaled_per_image = kl_div_per_image / num_pixels
                
                final_anomaly_score = recon_loss_per_image + kl_scaled_per_image
                anomaly_scores.extend(final_anomaly_score.cpu().numpy())
                
            elif args.model == 'sae':
                encoded = encoder(batch_x)
                reconstructed = decoder(encoded)
                
                mse_per_pixel = F.mse_loss(reconstructed, batch_x, reduction='none')
                mse_per_image = mse_per_pixel.view(mse_per_pixel.size(0), -1).mean(dim=1)
                anomaly_scores.extend(mse_per_image.cpu().numpy())
                
            elif args.model == 'ae':
                encoded = encoder(batch_x)
                reconstructed = decoder(encoded)
                
                loss_per_pixel = mse_loss_fn(reconstructed, batch_x)
                loss_per_image = loss_per_pixel.view(loss_per_pixel.size(0), -1).mean(dim=1)
                anomaly_scores.extend(loss_per_image.cpu().numpy())
                latent_vectors.extend(encoded.view(encoded.size(0), -1).cpu().numpy())

            elif args.model == 'hybrid':
                z, p = encoder(batch_x)
                reconstructed = decoder(z)
                
                loss_per_pixel = mse_loss_fn(reconstructed, batch_x)
                loss_per_image = loss_per_pixel.view(loss_per_pixel.size(0), -1).mean(dim=1)
                anomaly_scores.extend(loss_per_image.cpu().numpy())
                
            true_labels.extend(batch_y.numpy())

    anomaly_scores = np.array(anomaly_scores)
    true_labels = np.array(true_labels)
    if args.model == 'ae':
        latent_vectors = np.array(latent_vectors)
    
    mean_loss = np.mean(anomaly_scores)
    print(f"\nResults {data_split.upper()}:")
    print(f"Mean Reconstruction Loss (Anomaly Score): {mean_loss:.6f}")

    # PLOT 1: Anomaly Score Distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores[true_labels == 0], color='blue', label='Background', 
                 kde=True, stat='density', alpha=0.5, bins=50)
    
    if np.sum(true_labels == 1) > 0:
        sns.histplot(anomaly_scores[true_labels == 1], color='red', label='Anomalies', 
                     kde=True, stat='density', alpha=0.5, bins=50)
        
    plt.xlabel('Reconstruction Error (Anomaly Score)')
    plt.ylabel('Density')
    plt.title(f'Score Distribution - {data_split.capitalize()} ({model_name.upper()})')
    plt.legend()
    
    dist_path = os.path.join(save_dir, f'loss_dist_{model_name}_{data_split}.png')
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()

    # PLOT 2: ROC Curve
    roc_auc = None
    if np.sum(true_labels == 1) > 0:
        fpr, tpr, thresholds = roc_curve(true_labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {data_split.capitalize()} ({model_name.upper()})')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(save_dir, f'roc_curve_{model_name}_{data_split}.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        
        if args.model == 'ae':
            plt.figure(figsize=(8, 8))
            valid_idx = fpr > 0
            fpr_valid = fpr[valid_idx]
            tpr_valid = tpr[valid_idx]
            rejection = 1.0 / fpr_valid
            
            plt.plot(tpr_valid, rejection, color='purple', lw=2, label=f'Autoencoder Rejection')
            plt.yscale('log') 
            plt.xlim((0.0, 1.0))
            plt.xlabel('Signal Efficiency ($\\epsilon_S$)')
            plt.ylabel('Background Rejection ($1/\\epsilon_B$)')
            plt.title(f'Background Rejection Curve - {data_split.capitalize()}')
            plt.grid(True, which="both", ls="--", alpha=0.5)
            plt.legend(loc="upper right")
            
            rej_path = os.path.join(save_dir, f'rejection_curve_{model_name}_{data_split}.png')
            plt.savefig(rej_path, bbox_inches='tight')
            plt.close()
            
    else:
        print(f"Skipping ROC curves for {data_split.upper()} (No anomalies).")

    if args.model == 'ae' and len(latent_vectors) > 0:
        print(f"Computing PCA and t-SNE for latent space projection ({data_split.upper()})")
        label_names = {0: 'Background (QCD/Light)', 1: 'Anomalies (New Physics)'}
        mapped_labels = [label_names[l] for l in true_labels]

        # PCA 
        pca = PCA(n_components=2)
        latent_pca = pca.fit_transform(latent_vectors)
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=latent_pca[:, 0], y=latent_pca[:, 1], hue=mapped_labels, 
                        palette={'Background (QCD/Light)': 'blue', 'Anomalies (New Physics)': 'red'}, 
                        alpha=0.6)
        plt.title(f'PCA Latent Space Projection - {data_split.capitalize()}')
        plt.legend()
        pca_path = os.path.join(save_dir, f'pca_latent_{model_name}_{data_split}.png')
        plt.savefig(pca_path, bbox_inches='tight')
        plt.close()

        # t-SNE 
        tsne = TSNE(n_components=2, random_state=42)
        latent_tsne = tsne.fit_transform(latent_vectors)
        plt.figure(figsize=(8, 8))
        sns.scatterplot(x=latent_tsne[:, 0], y=latent_tsne[:, 1], hue=mapped_labels, 
                        palette={'Background (QCD/Light)': 'blue', 'Anomalies (New Physics)': 'red'}, 
                        alpha=0.6)
        plt.title(f't-SNE Latent Space Projection - {data_split.capitalize()}')
        plt.legend()
        tsne_path = os.path.join(save_dir, f'tsne_latent_{model_name}_{data_split}.png')
        plt.savefig(tsne_path, bbox_inches='tight')
        plt.close()

    return mean_loss, roc_auc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    if args.model == 'hybrid':
        from dataset.dataloader_supcon import get_dataloaders
        _, valid_loader, test_loader = get_dataloaders(
            data_filepath = args.data_path, 
            bg_classes = args.bg_classes,
            img_size = args.img_size, 
            batch_size = args.batch_size, 
            num_workers = min(4, os.cpu_count() or 1),
            max_samples = args.max_samples,
            binary_train_labels = False
        )
    else:
        from dataset.dataloader import get_dataloaders
        _, valid_loader, test_loader = get_dataloaders(
            data_filepath = args.data_path, 
            bg_classes = args.bg_classes,
            img_size = args.img_size, 
            batch_size = args.batch_size, 
            num_workers = min(4, os.cpu_count() or 1),
            max_samples = args.max_samples
        )
    
    if args.model == 'vae':
        from model.other_models_attempt.miniVAE import Encoder, Decoder
        encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    elif args.model == 'hybrid':
        from model.other_models_attempt.autoencoder import HybridEncoder as Encoder, Decoder
        encoder = Encoder(latent_space_dim=args.latent_space_dim, proj_dim=64).to(device)
    else:
        from model.other_models_attempt.autoencoder import Encoder, Decoder
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

    evaluate_unified(valid_loader, encoder, decoder, device, args.save_dir, args.model, "validation", args)
    evaluate_unified(test_loader, encoder, decoder, device, args.save_dir, args.model, "test", args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified Evaluation Script")
    parser.add_argument('--model', type=str, required=True, choices=['ae', 'vae', 'sae', 'hybrid'])
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./results_ad')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--latent_space_dim', type=int, default=128)
    args = parser.parse_args()
    main(args)