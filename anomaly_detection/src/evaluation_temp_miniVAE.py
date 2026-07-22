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
from model.other_models_attempt.miniVAE import Encoder, Decoder

def evaluate_anomaly_detection(dataloader, encoder, decoder, device, save_dir, model_name, data_split, sigma=1.0):
    encoder.eval()
    decoder.eval()

    mse_loss_fn = nn.MSELoss(reduction='none') 
    
    anomaly_scores = []
    true_labels = []

    print(f"\n--- Eval on set: {data_split.upper()} ---")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            
            # 1. NORMALIZZAZIONE DEI DATI
            if batch_x.max() > 1.0:
                batch_x = batch_x / 255.0
            
            # 2. FORWARD PASS (Gestione multipla)
            encoded, mu, log_var = encoder(batch_x)
            
            # 3. DETERMINISMO: Passiamo 'mu' al decoder
            reconstructed = decoder(mu)

            # 4. CALCOLO ANOMALY SCORE
            # A) Errore di ricostruzione (MSE) scalato per sigma^2
            loss_per_pixel = mse_loss_fn(reconstructed, batch_x)
            recon_loss_per_image = loss_per_pixel.view(loss_per_pixel.size(0), -1).mean(dim=1) / (sigma**2)

            # B) Divergenza KL
            kl_div_per_image = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
            
            # C) Scaliamo la KL come fatto nel training
            num_pixels = batch_x.shape[1] * batch_x.shape[2] * batch_x.shape[3]
            kl_scaled_per_image = kl_div_per_image / num_pixels

            # D) Somma bilanciata esattamente come nel training
            final_anomaly_score = recon_loss_per_image + kl_scaled_per_image

            # Salva i punteggi e le label vere
            anomaly_scores.extend(final_anomaly_score.cpu().numpy())
            true_labels.extend(batch_y.numpy())

    anomaly_scores = np.array(anomaly_scores)
    true_labels = np.array(true_labels)
    
    mean_loss = np.mean(anomaly_scores)
    print(f"\nResults {data_split.upper()}:")
    print(f"Mean Reconstruction Loss: {mean_loss:.6f}")

    # PLOT 1: Anomaly Score Distribution
    plt.figure(figsize=(10, 6))
    
    # Background (Label 0)
    sns.histplot(anomaly_scores[true_labels == 0], color='blue', label='Background (QCD/Light)', 
                 kde=True, stat='density', alpha=0.5, bins=50)
    
    # Anomalies (Label 1)
    if np.sum(true_labels == 1) > 0:
        sns.histplot(anomaly_scores[true_labels == 1], color='red', label='Anomalies (New Physics)', 
                     kde=True, stat='density', alpha=0.5, bins=50)
        
    plt.xlabel('Reconstruction Error (Anomaly Score)')
    plt.ylabel('Density')
    plt.title(f'Anomaly Score Distribution - {data_split.capitalize()}')
    plt.legend()
    
    dist_path = os.path.join(save_dir, f'loss_dist_{data_split}_{model_name}.png')
    plt.savefig(dist_path, bbox_inches='tight')
    plt.close()
    print(f"Distribution plot saved in: {dist_path}")

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
        
        plt.xlabel('False Positive Rate (Background Mistag)')
        plt.ylabel('True Positive Rate (Anomaly Efficiency)')
        plt.title(f'Anomaly Detection ROC Curve - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(save_dir, f'roc_curve_{data_split}_{model_name}.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        print(f"ROC plot saved in: {roc_path}")
    else:
        print(f"Skipping ROC curve for {data_split.upper()} (No anomalies present in validation set).")

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
    
    # Initialize the Variational Autoencoder model
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
                               save_dir=args.save_dir, model_name="variational_autoencoder")
    
    # Evaluate on test
    evaluate_anomaly_detection(test_loader, encoder, decoder, device, data_split="test", 
                               save_dir=args.save_dir, model_name="variational_autoencoder")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluation of Variational Autoencoder for Anomaly Detection")
    parser.add_argument('--model_path', type=str, required=True, help="Variational Autoencoder weights path")
    parser.add_argument('--data_path', type=str, default='./dataset.h5', help="Path to the dataset")
    parser.add_argument('--save_dir', type=str, default='./results_ad', help="Directory for plots and results")
    parser.add_argument('--max_samples', type=int, default=None, help="Maximum number of samples to use")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128, help='Image size')
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1], help='Classes to consider as background (e.g. 0 1)')
    parser.add_argument('--latent_space_dim', type=int, default=128, help='Dimension of the latent space')
    args = parser.parse_args()
    main(args)