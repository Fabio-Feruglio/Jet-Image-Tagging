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

def save_reconstruction_pairs_by_class(original, reconstructed, labels, save_dir, data_split, model_name, num_per_class=3):
    img_dir = os.path.join(save_dir, 'reconstructions')
    os.makedirs(img_dir, exist_ok=True)
    orig_np, recon_np, labels_np = original.cpu().detach().numpy(), reconstructed.cpu().detach().numpy(), labels.cpu().detach().numpy()
    
    bg_indices = np.where(labels_np == 0)[0][:num_per_class]
    anom_indices = np.where(labels_np == 1)[0][:num_per_class]
    selected_indices = np.concatenate([bg_indices, anom_indices])
    
    if len(selected_indices) == 0: return

    fig, axes = plt.subplots(nrows=len(selected_indices), ncols=2, figsize=(8, 3 * len(selected_indices)))
    if len(selected_indices) == 1: axes = [axes]
        
    for i, idx in enumerate(selected_indices):
        label_type = "Background" if labels_np[idx] == 0 else "Anomaly"
        img_in, img_out = orig_np[idx].squeeze(), recon_np[idx].squeeze()

        axes[i][0].imshow(img_in, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][0].set_title(f"Input - {label_type}")
        axes[i][0].axis('off')
        
        axes[i][1].imshow(img_out, cmap='gray', vmin=0.0, vmax=1.0)
        axes[i][1].set_title(f"{model_name} Recon - {label_type}")
        axes[i][1].axis('off')
        
    plt.tight_layout()
    plt.savefig(os.path.join(img_dir, f"recon_{model_name}_{data_split}.png"), bbox_inches='tight')
    plt.close(fig)

def evaluate_dae(dataloader, encoder, decoder, device, save_dir, data_split):
    encoder.eval()
    decoder.eval()
    mse_scores, true_labels = [], []
    saved_images = False

    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc=f"Evaluating {data_split}"):
            batch_x = batch_x.to(device)
            encoded = encoder(batch_x)
            reconstructed = decoder(encoded)

            if not saved_images and (batch_y == 0).any() and (batch_y == 1).any():
                save_reconstruction_pairs_by_class(batch_x, reconstructed, batch_y, save_dir, data_split, "DAE")
                saved_images = True

            mse_per_pixel = F.mse_loss(reconstructed, batch_x, reduction='none')
            mse_per_image = mse_per_pixel.view(mse_per_pixel.size(0), -1).mean(dim=1)
            
            mse_scores.extend(mse_per_image.cpu().numpy())
            true_labels.extend(batch_y.numpy())

    mse_scores, true_labels = np.array(mse_scores), np.array(true_labels)
    
    if np.sum(true_labels == 1) > 0:
        plt.figure(figsize=(10, 6))
        sns.histplot(mse_scores[true_labels == 0], color='blue', label='Background', kde=True, stat='density', alpha=0.5, bins=50)
        sns.histplot(mse_scores[true_labels == 1], color='red', label='Anomalies', kde=True, stat='density', alpha=0.5, bins=50)
        plt.xlabel('Reconstruction Error (MSE)')
        plt.ylabel('Density')
        plt.title(f'DAE Score Dist - {data_split.capitalize()}')
        plt.legend()
        plt.savefig(os.path.join(save_dir, f'loss_dist_dae_{data_split}.png'), bbox_inches='tight')
        plt.close()

        fpr, tpr, _ = roc_curve(true_labels, mse_scores)
        roc_auc = auc(fpr, tpr)
        plt.figure(figsize=(8, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
        plt.title(f'DAE ROC - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, f'roc_dae_{data_split}.png'), bbox_inches='tight')
        plt.close()

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)
    _, valid_loader, test_loader = get_dataloaders(
        data_filepath=args.data_path, bg_classes=args.bg_classes,
        img_size=args.img_size, batch_size=args.batch_size, max_samples=args.max_samples
    )
    encoder = Encoder(latent_space_dim=args.latent_space_dim).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    checkpoint = torch.load(args.model_path, map_location=device)
    encoder.load_state_dict(checkpoint.get('encoder_state_dict', checkpoint))
    decoder.load_state_dict(checkpoint.get('decoder_state_dict', checkpoint))

    evaluate_dae(valid_loader, encoder, decoder, device, args.save_dir, "validation")
    evaluate_dae(test_loader, encoder, decoder, device, args.save_dir, "test")

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