import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest

from dataset.dataloader import get_dataloaders 
from model.other_models_attempt.variational_autoencoder_advanced import Encoder, Decoder

def evaluate_advanced(dataloader, encoder, decoder, device, save_dir, data_split):
    encoder.eval()
    decoder.eval()
    mse_loss_fn = nn.MSELoss(reduction='none') 
    
    mse_scores, kl_scores, mu_vectors, true_labels = [], [], [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc=f"Evaluating {data_split}"):
            batch_x = batch_x.to(device)
            
            # Forward deterministico per la valutazione
            _, mu, log_var = encoder(batch_x)
            reconstructed = decoder(mu) # Passiamo mu per eliminare il rumore

            # Score 1: MSE (Reconstruction)
            loss_per_pixel = mse_loss_fn(reconstructed, batch_x)
            mse_per_image = loss_per_pixel.view(loss_per_pixel.size(0), -1).mean(dim=1)
            
            # Score 2: KL Divergence (Out-of-Distribution)
            kl_per_image = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)

            mse_scores.extend(mse_per_image.cpu().numpy())
            kl_scores.extend(kl_per_image.cpu().numpy())
            mu_vectors.extend(mu.cpu().numpy())
            true_labels.extend(batch_y.numpy())

    mse_scores = np.array(mse_scores)
    kl_scores = np.array(kl_scores)
    mu_vectors = np.array(mu_vectors)
    true_labels = np.array(true_labels)
    
    # Score 3: Density Estimation nello Spazio Latente (Isolation Forest)
    print("Addestramento Isolation Forest sui vettori latenti...")
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(mu_vectors) 
    
    # Restituisce score negativi (più negativo = più anomalo). Li invertiamo.
    if_scores = -iso_forest.decision_function(mu_vectors) 

    if np.sum(true_labels == 1) > 0:
        # --- 1. PLOT DELLE DISTRIBUZIONI (ISTOGRAMMI) ---
        metrics = {
            'Reconstruction_MSE': mse_scores,
            'KL_Divergence': kl_scores,
            'Latent_Density_IsoForest': if_scores
        }

        for metric_name, scores in metrics.items():
            plt.figure(figsize=(10, 6))
            
            # Background (Label 0)
            sns.histplot(scores[true_labels == 0], color='blue', label='Background (QCD/Light)', 
                         kde=True, stat='density', alpha=0.5, bins=50)
            # Anomalies (Label 1)
            sns.histplot(scores[true_labels == 1], color='red', label='Anomalies (New Physics)', 
                         kde=True, stat='density', alpha=0.5, bins=50)
                
            plt.xlabel(f'{metric_name.replace("_", " ")} Score')
            plt.ylabel('Density')
            plt.title(f'{metric_name.replace("_", " ")} Distribution - {data_split.capitalize()}')
            plt.legend()
            
            dist_path = os.path.join(save_dir, f'loss_dist_{metric_name}_{data_split}.png')
            plt.savefig(dist_path, bbox_inches='tight')
            plt.close()
            print(f"Istogramma salvato in: {dist_path}")

        # --- 2. PLOT ROC COMPARATIVO ---
        plt.figure(figsize=(10, 8))
        
        # Curve per MSE
        fpr_mse, tpr_mse, _ = roc_curve(true_labels, mse_scores)
        plt.plot(fpr_mse, tpr_mse, lw=2, label=f'Reconstruction (MSE) AUC = {auc(fpr_mse, tpr_mse):.3f}')
        
        # Curve per KL
        fpr_kl, tpr_kl, _ = roc_curve(true_labels, kl_scores)
        plt.plot(fpr_kl, tpr_kl, lw=2, label=f'KL Divergence AUC = {auc(fpr_kl, tpr_kl):.3f}')
        
        # Curve per Isolation Forest su mu
        fpr_if, tpr_if, _ = roc_curve(true_labels, if_scores)
        plt.plot(fpr_if, tpr_if, lw=2, label=f'Latent Density (IsoForest) AUC = {auc(fpr_if, tpr_if):.3f}')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Background Mistag)')
        plt.ylabel('True Positive Rate (Anomaly Efficiency)')
        plt.title(f'Multi-Metric Anomaly Detection ROC - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(save_dir, f'roc_comparison_{data_split}.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        print(f"ROC comparativa salvata in: {roc_path}")
    else:
        print(f"Nessuna anomalia trovata in {data_split}. Skippo i plot ROC e distribuzioni.")
        
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load dataloaders
    _, valid_loader, test_loader = get_dataloaders(
        data_filepath=args.data_path, bg_classes=args.bg_classes,
        img_size=args.img_size, batch_size=args.batch_size, max_samples=args.max_samples
    )
    
    # Initialize the Variational Autoencoder model
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    
    print(f"Loading model weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
        decoder.load_state_dict(checkpoint)

    # Valutazione
    evaluate_advanced(test_loader, encoder, decoder, device, args.save_dir, data_split="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help="Path ai pesi del VAE addestrato")
    parser.add_argument('--data_path', type=str, default='./dataset.h5', help="Path al dataset")
    parser.add_argument('--save_dir', type=str, default='./results_ad_advanced', help="Cartella per i grafici")
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--latent_space_dim', type=int, default=32)
    args = parser.parse_args()
    main(args)