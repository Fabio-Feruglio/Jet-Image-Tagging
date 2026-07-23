import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import IsolationForest

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.hybrid_ae_supcon import HybridEncoder, Decoder

def evaluate_hybrid(dataloader, encoder, decoder, device, save_dir, data_split):
    encoder.eval()
    decoder.eval()
    
    mse_scores, p_vectors, true_labels = [], [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc=f"Evaluating {data_split}"):
            batch_x = batch_x.to(device)
            
            # Estraiamo z (per la ricostruzione) e p (la proiezione contrastiva L2)
            z, p = encoder(batch_x)
            reconstructed = decoder(z)

            # Score 1: MSE per immagine
            mse_per_pixel = F.mse_loss(reconstructed, batch_x, reduction='none')
            mse_per_image = mse_per_pixel.view(mse_per_pixel.size(0), -1).mean(dim=1)
            
            mse_scores.extend(mse_per_image.cpu().numpy())
            p_vectors.extend(p.cpu().numpy())
            true_labels.extend(batch_y.numpy())

    mse_scores = np.array(mse_scores)
    p_vectors = np.array(p_vectors)
    true_labels = np.array(true_labels)
    
    # Score 2: Anomaly detection sui vettori proiettati 'p'
    print("Addestramento Isolation Forest sui vettori proiettati...")
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(p_vectors)
    # Inverto i valori: score negativi indicano anomalie (outliers) in sklearn
    if_scores = -iso_forest.decision_function(p_vectors)

    if np.sum(true_labels == 1) > 0:
        metrics = {
            'Reconstruction_MSE': mse_scores,
            'Contrastive_Space_IsolationForest': if_scores
        }

        # --- PLOT DELLE DISTRIBUZIONI ---
        for metric_name, scores in metrics.items():
            plt.figure(figsize=(10, 6))
            sns.histplot(scores[true_labels == 0], color='blue', label='Background (QCD/Light)', 
                         kde=True, stat='density', alpha=0.5, bins=50)
            sns.histplot(scores[true_labels == 1], color='red', label='Anomalies (New Physics)', 
                         kde=True, stat='density', alpha=0.5, bins=50)
                
            plt.xlabel(f'{metric_name.replace("_", " ")} Score')
            plt.ylabel('Density')
            plt.title(f'{metric_name.replace("_", " ")} - {data_split.capitalize()}')
            plt.legend()
            
            dist_path = os.path.join(save_dir, f'loss_dist_{metric_name}_{data_split}.png')
            plt.savefig(dist_path, bbox_inches='tight')
            plt.close()
            print(f"Istogramma salvato in: {dist_path}")

        # --- PLOT ROC COMPARATIVO ---
        plt.figure(figsize=(10, 8))
        
        # Curva MSE
        fpr_mse, tpr_mse, _ = roc_curve(true_labels, mse_scores)
        plt.plot(fpr_mse, tpr_mse, lw=2, label=f'Reconstruction (MSE) AUC = {auc(fpr_mse, tpr_mse):.3f}')
        
        # Curva Contrastiva (Isolation Forest)
        fpr_if, tpr_if, _ = roc_curve(true_labels, if_scores)
        plt.plot(fpr_if, tpr_if, lw=2, label=f'Contrastive Space (IsoForest) AUC = {auc(fpr_if, tpr_if):.3f}')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Background Mistag)')
        plt.ylabel('True Positive Rate (Anomaly Efficiency)')
        plt.title(f'Hybrid SupCon Anomaly Detection ROC - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(save_dir, f'roc_hybrid_comparison_{data_split}.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        print(f"ROC comparativa salvata in: {roc_path}")
    else:
        print(f"Nessuna anomalia in {data_split}. Skip plot.")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    _, _, test_loader = get_dataloaders(
        data_filepath=args.data_path, bg_classes=args.bg_classes,
        img_size=args.img_size, batch_size=args.batch_size, max_samples=args.max_samples
    )
    
    encoder = HybridEncoder(latent_space_dim=args.latent_space_dim, proj_dim=64).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    evaluate_hybrid(test_loader, encoder, decoder, device, args.save_dir, data_split="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./results_hybrid_supcon')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--latent_space_dim', type=int, default=128)
    args = parser.parse_args()
    main(args)