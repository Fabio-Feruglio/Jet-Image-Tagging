import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from dataset.dataloader import get_dataloaders
from model.other_models_attempt.autoencoder import Encoder, Decoder

def evaluate_dae(dataloader, encoder, decoder, device, save_dir, data_split):
    encoder.eval()
    decoder.eval()
    
    mse_scores, true_labels = [], []

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc=f"Evaluating {data_split}"):
            batch_x = batch_x.to(device)
            
            # Inferenza pulita (niente rumore)
            encoded = encoder(batch_x)
            reconstructed = decoder(encoded)

            # Usiamo l'MSE standard per l'Anomaly Score (media su tutti i pixel dell'immagine)
            mse_per_pixel = F.mse_loss(reconstructed, batch_x, reduction='none')
            mse_per_image = mse_per_pixel.view(mse_per_pixel.size(0), -1).mean(dim=1)
            
            mse_scores.extend(mse_per_image.cpu().numpy())
            true_labels.extend(batch_y.numpy())

    mse_scores = np.array(mse_scores)
    true_labels = np.array(true_labels)
    
    if np.sum(true_labels == 1) > 0:
        # --- PLOT DISTRIBUZIONE ---
        plt.figure(figsize=(10, 6))
        sns.histplot(mse_scores[true_labels == 0], color='blue', label='Background (QCD/Light)', 
                     kde=True, stat='density', alpha=0.5, bins=50)
        sns.histplot(mse_scores[true_labels == 1], color='red', label='Anomalies (New Physics)', 
                     kde=True, stat='density', alpha=0.5, bins=50)
            
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Density')
        plt.title(f'DAE Anomaly Score Distribution - {data_split.capitalize()}')
        plt.legend()
        
        dist_path = os.path.join(save_dir, f'loss_dist_dae_{data_split}.png')
        plt.savefig(dist_path, bbox_inches='tight')
        plt.close()
        print(f"Istogramma salvato in: {dist_path}")

        # --- PLOT ROC ---
        fpr, tpr, _ = roc_curve(true_labels, mse_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate (Background Mistag)')
        plt.ylabel('True Positive Rate (Anomaly Efficiency)')
        plt.title(f'DAE Anomaly Detection ROC - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        
        roc_path = os.path.join(save_dir, f'roc_dae_{data_split}.png')
        plt.savefig(roc_path, bbox_inches='tight')
        plt.close()
        print(f"ROC salvata in: {roc_path}")
    else:
        print(f"Nessuna anomalia in {data_split}. Skip plot.")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    
    _, _, test_loader = get_dataloaders(
        data_filepath=args.data_path, bg_classes=args.bg_classes,
        img_size=args.img_size, batch_size=args.batch_size, max_samples=args.max_samples
    )
    
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    
    checkpoint = torch.load(args.model_path, map_location=device)
    encoder.load_state_dict(checkpoint['encoder_state_dict'])
    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    evaluate_dae(test_loader, encoder, decoder, device, args.save_dir, data_split="test")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./results_dae')
    parser.add_argument('--max_samples', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--latent_space_dim', type=int, default=128)
    args = parser.parse_args()
    main(args)