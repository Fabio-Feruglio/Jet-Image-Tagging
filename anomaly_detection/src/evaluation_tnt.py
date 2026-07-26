import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

from dataset.dataloader_tnt import get_dataloaders_tnt 
from model.other_models_attempt.autoencoder import Encoder, Decoder

# --- REINTEGRATO: Salvataggio Ricostruzioni Immagini ---
def save_reconstruction_pairs_by_class(original, reconstructed, labels, save_dir, model_name, num_per_class=3):
    img_dir = os.path.join(save_dir, 'reconstructions')
    os.makedirs(img_dir, exist_ok=True)
    
    orig_np = original.cpu().detach().numpy()
    recon_np = reconstructed.cpu().detach().numpy()
    labels_np = labels.cpu().detach().numpy()
    
    bg_indices = np.where(labels_np == 0)[0][:num_per_class]
    anom_indices = np.where(labels_np == 1)[0][:num_per_class]
    selected_indices = np.concatenate([bg_indices, anom_indices])
    total_images = len(selected_indices)
    
    if total_images == 0:
        return

    fig, axes = plt.subplots(nrows=total_images, ncols=2, figsize=(8, 3 * total_images))
    if total_images == 1:
        axes = [axes]
        
    for i, idx in enumerate(selected_indices):
        label_type = "Background" if labels_np[idx] == 0 else "Anomaly"
        img_in = orig_np[idx].squeeze()
        img_out = recon_np[idx].squeeze()

        axes[i][0].imshow(img_in, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][0].set_title(f"Input - {label_type}")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(img_out, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][1].set_title(f"{model_name} Recon - {label_type}")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    save_path = os.path.join(img_dir, f"reconstructions_comparison_{model_name}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
# -------------------------------------------------------

# --- REINTEGRATO: Proiezioni PCA e t-SNE ---
def plot_latent_space(latent_vectors, true_labels, save_dir, model_name):
    print(f"Computing PCA and t-SNE per il modello {model_name}...")
    label_names = {0: 'Background', 1: 'Anomalies'}
    mapped_labels = [label_names[l] for l in true_labels]

    pca = PCA(n_components=2)
    latent_pca = pca.fit_transform(latent_vectors)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latent_pca[:, 0], y=latent_pca[:, 1], hue=mapped_labels, 
                    palette={'Background': 'blue', 'Anomalies': 'red'}, alpha=0.6)
    plt.title(f'PCA Latent Space - {model_name}')
    plt.savefig(os.path.join(save_dir, f'pca_latent_{model_name}.png'), bbox_inches='tight')
    plt.close()

    tsne = TSNE(n_components=2, random_state=42)
    latent_tsne = tsne.fit_transform(latent_vectors)
    
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=latent_tsne[:, 0], y=latent_tsne[:, 1], hue=mapped_labels, 
                    palette={'Background': 'blue', 'Anomalies': 'red'}, alpha=0.6)
    plt.title(f't-SNE Latent Space - {model_name}')
    plt.savefig(os.path.join(save_dir, f'tsne_latent_{model_name}.png'), bbox_inches='tight')
    plt.close()
# -------------------------------------------

def evaluate_2d_anomaly(dataloader, enc1, dec1, enc2, dec2, device, save_dir):
    enc1.eval(); dec1.eval()
    enc2.eval(); dec2.eval()
    mse_fn = nn.MSELoss(reduction='none') 
    
    loss_bkg_list, loss_sig_list, labels_list = [], [], []
    latent_ae1_list, latent_ae2_list = [], []
    
    saved_images = False

    print("\n=== Valutazione sull'Evaluation Set ===")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader):
            batch_x = batch_x.to(device)
            
            # Loss e Latent AE1 (Background)
            encoded1 = enc1(batch_x)
            rec1 = dec1(encoded1)
            l1 = mse_fn(rec1, batch_x).view(batch_x.size(0), -1).mean(dim=1)
            
            # Loss e Latent AE2 (Pseudo-Segnale)
            encoded2 = enc2(batch_x)
            rec2 = dec2(encoded2)
            l2 = mse_fn(rec2, batch_x).view(batch_x.size(0), -1).mean(dim=1)
            
            # Salvataggio Immagini al primo batch utile
            if not saved_images:
                if (batch_y == 0).any() and (batch_y == 1).any():
                    save_reconstruction_pairs_by_class(batch_x, rec1, batch_y, save_dir, "ae1_bkg")
                    save_reconstruction_pairs_by_class(batch_x, rec2, batch_y, save_dir, "ae2_sig")
                    saved_images = True
            
            loss_bkg_list.extend(l1.cpu().numpy())
            loss_sig_list.extend(l2.cpu().numpy())
            labels_list.extend(batch_y.numpy())
            
            latent_ae1_list.extend(encoded1.view(encoded1.size(0), -1).cpu().numpy())
            latent_ae2_list.extend(encoded2.view(encoded2.size(0), -1).cpu().numpy())

    loss_bkg = np.array(loss_bkg_list)
    loss_sig = np.array(loss_sig_list)
    labels = np.array(labels_list)
    latent_ae1 = np.array(latent_ae1_list)
    latent_ae2 = np.array(latent_ae2_list)
    
    # Proiezioni degli Spazi Latenti (Reintegrate)
    plot_latent_space(latent_ae1, labels, save_dir, "ae1_bkg")
    plot_latent_space(latent_ae2, labels, save_dir, "ae2_sig")
    
    # Calcolo Score Ibrido
    hybrid_score = loss_bkg / (loss_sig + 1e-8)
    os.makedirs(save_dir, exist_ok=True)
    
    # PLOT 1: Spazio Latente 2D delle Loss
    plt.figure(figsize=(8, 8))
    sns.scatterplot(x=loss_bkg[labels==0], y=loss_sig[labels==0], color='blue', label='Background', alpha=0.3)
    if np.sum(labels == 1) > 0:
        sns.scatterplot(x=loss_bkg[labels==1], y=loss_sig[labels==1], color='red', label='Anomalies', alpha=0.6)
    plt.xlabel('Reconstruction Error - AE1 (Background Model)')
    plt.ylabel('Reconstruction Error - AE2 (Pseudo-Anomaly Model)')
    plt.title('2D QUAK/Tag N\' Train Loss Space')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_space_2d.png'), bbox_inches='tight')
    plt.close()

    # PLOT 2: Distribuzione Score Ibrido
    plt.figure(figsize=(10, 6))
    sns.histplot(hybrid_score[labels == 0], color='blue', label='Background', kde=True, stat='density', alpha=0.5, bins=50)
    if np.sum(labels == 1) > 0:
        sns.histplot(hybrid_score[labels == 1], color='red', label='Anomalies', kde=True, stat='density', alpha=0.5, bins=50)
    plt.xlabel('Hybrid Anomaly Score (MSE_Bkg / MSE_Sig)')
    plt.ylabel('Density')
    plt.title('Hybrid Score Distribution')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'hybrid_score_dist.png'), bbox_inches='tight')
    plt.close()

    # PLOT 3: Curva ROC e AUC
    if np.sum(labels == 1) > 0:
        fpr, tpr, _ = roc_curve(labels, hybrid_score)
        roc_auc = auc(fpr, tpr)
        
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'Hybrid Score ROC (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, 'roc_hybrid.png'), bbox_inches='tight')
        plt.close()
        
        print(f"\nFinal Hybrid AUC: {roc_auc:.4f}")

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, _, eval_loader = get_dataloaders_tnt(
        args.data_path, args.bg_classes, args.img_size, args.batch_size, 0, args.max_samples
    )
    
    enc1 = Encoder(args.latent_space_dim).to(device)
    dec1 = Decoder(args.latent_space_dim).to(device)
    enc2 = Encoder(args.latent_space_dim).to(device)
    dec2 = Decoder(args.latent_space_dim).to(device)
    
    ckpt1 = torch.load(os.path.join(args.model_dir, 'ae1_bkg_best.pth'), map_location=device)
    enc1.load_state_dict(ckpt1['encoder'])
    dec1.load_state_dict(ckpt1['decoder'])
    
    ckpt2 = torch.load(os.path.join(args.model_dir, 'ae2_sig_best.pth'), map_location=device)
    enc2.load_state_dict(ckpt2['encoder'])
    dec2.load_state_dict(ckpt2['decoder'])

    evaluate_2d_anomaly(eval_loader, enc1, dec1, enc2, dec2, device, args.save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, required=True, help="Directory containing ae1_bkg_best.pth and ae2_sig_best.pth")
    parser.add_argument('--data_path', type=str, default='./dataset.h5')
    parser.add_argument('--save_dir', type=str, default='./results_tnt')
    parser.add_argument('--max_samples', type=int, default=50000)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--latent_space_dim', type=int, default=16)
    main(parser.parse_args())