import argparse
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc

from dataset.dataloader_tnt import get_dataloaders_tnt 
from model.other_models_attempt.autoencoder import Encoder, Decoder

def evaluate_2d_anomaly(dataloader, enc1, dec1, enc2, dec2, device, save_dir):
    enc1.eval(); dec1.eval()
    enc2.eval(); dec2.eval()
    mse_fn = torch.nn.MSELoss(reduction='none') 
    
    loss_bkg_list, loss_sig_list, labels_list = [], [], []

    print("\n=== Valutazione sull'Evaluation Set ===")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader):
            batch_x = batch_x.to(device)
            
            # Loss AE1 (Background)
            rec1 = dec1(enc1(batch_x))
            l1 = mse_fn(rec1, batch_x).view(batch_x.size(0), -1).mean(dim=1)
            
            # Loss AE2 (Pseudo-Segnale)
            rec2 = dec2(enc2(batch_x))
            l2 = mse_fn(rec2, batch_x).view(batch_x.size(0), -1).mean(dim=1)
            
            loss_bkg_list.extend(l1.cpu().numpy())
            loss_sig_list.extend(l2.cpu().numpy())
            labels_list.extend(batch_y.numpy())

    loss_bkg = np.array(loss_bkg_list)
    loss_sig = np.array(loss_sig_list)
    labels = np.array(labels_list)
    
    # 1. Calcolo Score Ibrido (Rapporto tra le loss)
    # Evitiamo divisioni per zero aggiungendo un epsilon
    hybrid_score = loss_bkg / (loss_sig + 1e-8)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # 2. PLOT 1: Spazio Latente 2D delle Loss
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

    # 3. PLOT 2: Distribuzione Score Ibrido
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

    # 4. PLOT 3: Curva ROC e AUC
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
    parser.add_argument('--max_samples', type=int, default=30000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--img_size', type=int, default=128)
    parser.add_argument('--bg_classes', nargs='+', type=int, default=[0, 1])
    parser.add_argument('--latent_space_dim', type=int, default=16)
    main(parser.parse_args())