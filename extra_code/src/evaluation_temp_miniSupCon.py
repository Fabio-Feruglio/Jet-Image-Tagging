import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import seaborn as sns
from sklearn.metrics import roc_curve, auc

from dataset.dataloader import get_dataloaders 
from model.other_models_attempt.miniVAE import Encoder, Decoder, TransformHead

def save_reconstruction_pairs_by_class(original, reconstructed, labels, save_dir, data_split, num_per_class=3):
    img_dir = os.path.join(save_dir, 'reconstructions')
    os.makedirs(img_dir, exist_ok=True)
    
    orig_np = original.cpu().detach().numpy()
    recon_np = reconstructed.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    
    bg_indices = np.where(labels_np == 0)[0][:num_per_class]
    anom_indices = np.where(labels_np == 1)[0][:num_per_class]
    selected_indices = np.concatenate([bg_indices, anom_indices])
    total_images = len(selected_indices)
    
    if total_images == 0: return

    fig, axes = plt.subplots(nrows=total_images, ncols=2, figsize=(8, 3 * total_images))
    if total_images == 1: axes = [axes]
        
    for i, idx in enumerate(selected_indices):
        label_type = "Background" if labels_np[idx] == 0 else "Anomaly"
        img_in = orig_np[idx].squeeze()
        img_out = recon_np[idx].squeeze()

        axes[i][0].imshow(img_in, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][0].set_title(f"Input - {label_type}")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(img_out, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][1].set_title(f"Recon - {label_type}")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"reconstructions_comparison_{data_split}.png"), bbox_inches='tight')
    plt.close(fig)

def evaluate_anomaly_detection(dataloader, encoder, decoder, transform_head, device, save_dir, model_name, data_split, args):
    encoder.eval()
    decoder.eval()
    transform_head.eval()

    anomaly_scores = []
    true_labels = []
    saved_images = False

    # Parametro per i campionamenti multipli stocastici
    M_SAMPLES = 10 

    print(f"\n--- Eval on set: {data_split.upper()} ---")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            
            if batch_x.max() > 1.0:
                batch_x = batch_x / 255.0
            
            # Forward Encoder
            _, mu, log_var = encoder(batch_x)
            
            # 1. Calcolo Incertezza e Ricostruzione su M campionamenti stocastici
            recon_errors_per_sample = []
            
            # Creiamo la matrice dei pesi per la Weighted Loss
            weight = torch.where(batch_x > 0.0, 10.0, 1.0)
            
            # Salveremo la prima ricostruzione deterministica solo per il plot
            deterministic_recon = decoder(mu)
            
            for _ in range(M_SAMPLES):
                # Campionamento dal reparameterization trick
                std = torch.exp(0.5 * log_var)
                eps = torch.randn_like(std)
                z = mu + eps * std
                
                # Ricostruzione Stocastica
                rec_x = decoder(z)
                
                # Calcolo errore pesato per singola immagine [batch_size]
                err = (weight * (rec_x - batch_x)**2).view(batch_x.size(0), -1).mean(dim=1)
                recon_errors_per_sample.append(err)
            
            # Tensore degli errori: shape [M_SAMPLES, batch_size]
            recon_errors_tensor = torch.stack(recon_errors_per_sample, dim=0)
            
            # L'incertezza genera due segnali: l'errore medio e la varianza delle ricostruzioni
            mean_recon_error = recon_errors_tensor.mean(dim=0)
            var_recon_error = recon_errors_tensor.var(dim=0)

            # 2. Divergenza KL Scalata
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            num_pixels = batch_x.shape[1] * batch_x.shape[2] * batch_x.shape[3]
            kl_scaled = kl_div / num_pixels

            # 3. ANOMALY SCORE FINALE COMPLETO
            # Combina la ricostruzione media pesata, la varianza dell'incertezza e la KL
            final_anomaly_score = mean_recon_error + var_recon_error + (args.beta * kl_scaled)

            # --- Salvataggio Immagini Plot ---
            if not saved_images:
                has_bg = (batch_y == 0).any()
                has_anom = (batch_y == 1).any()
                if has_bg and has_anom:
                    save_reconstruction_pairs_by_class(
                        original=batch_x, 
                        reconstructed=deterministic_recon, 
                        labels=batch_y, 
                        save_dir=save_dir, 
                        data_split=data_split, 
                        num_per_class=3 
                    )
                    saved_images = True

            anomaly_scores.extend(final_anomaly_score.cpu().numpy())
            true_labels.extend(batch_y.numpy())

    anomaly_scores = np.array(anomaly_scores)
    true_labels = np.array(true_labels)
    
    # --- PLOTTING ---
    plt.figure(figsize=(10, 6))
    sns.histplot(anomaly_scores[true_labels == 0], color='blue', label='Background', kde=False, stat='density', alpha=0.5, bins=50)
    if np.sum(true_labels == 1) > 0:
        sns.histplot(anomaly_scores[true_labels == 1], color='red', label='Anomalies', kde=False, stat='density', alpha=0.5, bins=50)
    plt.xlabel('Advanced Anomaly Score (Weighted MSE + Var + Beta*KL)')
    plt.ylabel('Density')
    plt.title(f'Anomaly Score Distribution - {data_split.capitalize()}')
    plt.legend()
    dist_path = os.path.join(save_dir, f'loss_dist_{data_split}_{model_name}.png')
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()

    roc_auc = None
    if np.sum(true_labels == 1) > 0:
        fpr, tpr, _ = roc_curve(true_labels, anomaly_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'AUC = {roc_auc:.3f}')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim((0.0, 1.0))
        plt.ylim((0.0, 1.05))
        plt.xlabel('False Positive Rate (Background Mistag)')
        plt.ylabel('True Positive Rate (Anomaly Efficiency)')
        plt.title(f'ROC Curve - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        roc_path = os.path.join(save_dir, f'roc_curve_{data_split}_{model_name}.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        print(f"[{data_split.upper()}] Final AUC: {roc_auc:.4f}")
    
    return roc_auc

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    _, valid_loader, test_loader = get_dataloaders(
        data_filepath = args.data_path, 
        bg_classes = args.bg_classes,
        img_size = args.img_size, 
        batch_size = args.batch_size, 
        num_workers = min(4, os.cpu_count() or 1),
        max_samples = args.max_samples
    )
    
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    transform_head = TransformHead(latent_space_dim=args.latent_space_dim, proj_dim=args.proj_dim).to(device)
    
    print(f"Loading weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
        if 'transform_head_state_dict' in checkpoint:
            transform_head.load_state_dict(checkpoint['transform_head_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
        decoder.load_state_dict(checkpoint)

    evaluate_anomaly_detection(valid_loader, encoder, decoder, transform_head, device, args.save_dir, "advanced_vae", "validation", args)
    evaluate_anomaly_detection(test_loader, encoder, decoder, transform_head, device, args.save_dir, "advanced_vae", "test", args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced Evaluation of VAE for Anomaly Detection")
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./results_ad')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--latent_space_dim', type=int, default=128)
    
    # Parametri architettura
    parser.add_argument('--proj_dim', type=int, default=64)
    parser.add_argument('--beta', type=float, default=0.01, help='Beta per il bilanciamento KL, deve combaciare col training')
    
    args = parser.parse_args()
    main(args)