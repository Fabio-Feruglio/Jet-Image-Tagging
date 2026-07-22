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
from model.other_models_attempt.autoencoder import Encoder, Decoder

def save_reconstruction_pairs_by_class(original, reconstructed, labels, save_dir, data_split, num_per_class=3):
    """
    Salva un'immagine PNG contenente num_per_class immagini di Background e 
    num_per_class immagini Anomale, affiancate dalle loro ricostruzioni VAE.
    
    Format supportato: Singolo Canale (Grayscale) [Batch, 1, H, W] con dimensioni spaziali dinamiche.
    """
    img_dir = os.path.join(save_dir, 'reconstructions')
    os.makedirs(img_dir, exist_ok=True)
    
    # Trasformiamo i tensori in numpy array per il plotting
    orig_np = original.cpu().detach().numpy()
    recon_np = reconstructed.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    
    # Identifichiamo gli indici per background (0) e anomalie (1)
    bg_indices = np.where(labels_np == 0)[0][:num_per_class]
    anom_indices = np.where(labels_np == 1)[0][:num_per_class]
    
    # Uniamo gli indici trovati (prima tutti i background, poi tutte le anomalie)
    selected_indices = np.concatenate([bg_indices, anom_indices])
    total_images = len(selected_indices)
    
    if total_images == 0:
        print(f"Attenzione: Nessun dato trovato per salvare le ricostruzioni su {data_split}.")
        return

    # Creiamo la figura: 2 colonne (Input vs Output) x N righe totali
    fig, axes = plt.subplots(nrows=total_images, ncols=2, figsize=(8, 3 * total_images))
    
    # Gestione di sicurezza per indicizzazione se total_images == 1
    if total_images == 1:
        axes = [axes]
        
    for i, idx in enumerate(selected_indices):
        label_type = "Background (Normal)" if labels_np[idx] == 0 else "Anomaly (New Physics)"
        
        # Rimuoviamo la dimensione del canale (da [1, H, W] a [H, W]) usando squeeze()
        img_in = orig_np[idx].squeeze()
        img_out = recon_np[idx].squeeze()

        # Colonna 1: Input Originale con tipo di dato indicato nel titolo
        ax_orig = axes[i][0]
        ax_orig.imshow(img_in, cmap='gray', vmin=0.0, vmax=1.0)
        ax_orig.set_title(f"Input - {label_type}")
        ax_orig.axis('off')
        
        # Colonna 2: Output Ricostruito dal VAE
        ax_recon = axes[i][1]
        ax_recon.imshow(img_out, cmap='gray', vmin=0.0, vmax=1.0)
        ax_recon.set_title(f"VAE Recon - {label_type}")
        ax_recon.axis('off')
        
    plt.tight_layout()
    
    save_path = os.path.join(img_dir, f"reconstructions_comparison_{data_split}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Confronto ricostruzioni salvato in: {save_path}")

def evaluate_anomaly_detection(dataloader, encoder, decoder, device, save_dir, model_name, data_split):
    encoder.eval()
    decoder.eval()

    mse_loss_fn = nn.MSELoss(reduction='none') 
    
    anomaly_scores = []
    true_labels = []

    saved_images = False
    
    print(f"\n--- Eval on set: {data_split.upper()} ---")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            # batch_y contains the true labels (0 for background, 1 for anomaly)
            
            # Forward pass
            encoded = encoder(batch_x)
            reconstructed = decoder(encoded)

            if not saved_images:
                            # Verifichiamo che il batch contenga sia 0 che 1
                            has_bg = (batch_y == 0).any()
                            has_anom = (batch_y == 1).any()
                            
                            if has_bg and has_anom:
                                save_reconstruction_pairs_by_class(
                                    original=batch_x, 
                                    reconstructed=reconstructed, 
                                    labels=batch_y, 
                                    save_dir=save_dir, 
                                    data_split=data_split, 
                                    num_per_class=3 # Salverà 3 coppie di Background e 3 di Anomalie
                                )
                                saved_images = True

            # Anomaly score
            # shape [batch_size, channels, height, width]
            loss_per_pixel = mse_loss_fn(reconstructed, batch_x)
            
            # mean over channels, height, and width to get a single score per image
            # shape [batch_size]
            loss_per_image = loss_per_pixel.view(loss_per_pixel.size(0), -1).mean(dim=1)

            # Save the scores and true labels
            anomaly_scores.extend(loss_per_image.cpu().numpy())
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