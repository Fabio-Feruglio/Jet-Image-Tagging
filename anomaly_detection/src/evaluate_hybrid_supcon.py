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

def save_reconstruction_pairs_by_class(original, reconstructed, labels, save_dir, data_split, model_name, num_per_class=3):
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
        print(f"Attenzione: Nessun dato trovato per salvare le ricostruzioni su {data_split}.")
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
    save_path = os.path.join(img_dir, f"reconstructions_comparison_{model_name}_{data_split}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)

def evaluate_hybrid(dataloader, encoder, decoder, device, save_dir, data_split):
    encoder.eval()
    decoder.eval()
    mse_scores, p_vectors, true_labels = [], [], []
    saved_images = False

    print(f"\n--- Eval on set: {data_split.upper()} ---")
    with torch.no_grad():
        for batch_x, batch_y in tqdm(dataloader, desc="Evaluating"):
            batch_x = batch_x.to(device)
            
            z, p = encoder(batch_x)
            reconstructed = decoder(z)

            if not saved_images:
                has_bg = (batch_y == 0).any()
                has_anom = (batch_y == 1).any()
                if has_bg and has_anom:
                    save_reconstruction_pairs_by_class(
                        original=batch_x, reconstructed=reconstructed, labels=batch_y, 
                        save_dir=save_dir, data_split=data_split, model_name="Hybrid", num_per_class=3
                    )
                    saved_images = True

            mse_per_pixel = F.mse_loss(reconstructed, batch_x, reduction='none')
            mse_per_image = mse_per_pixel.view(mse_per_pixel.size(0), -1).mean(dim=1)
            
            mse_scores.extend(mse_per_image.cpu().numpy())
            p_vectors.extend(p.cpu().numpy())
            true_labels.extend(batch_y.numpy())

    mse_scores = np.array(mse_scores)
    p_vectors = np.array(p_vectors)
    true_labels = np.array(true_labels)
    
    print("Addestramento Isolation Forest sui vettori proiettati...")
    iso_forest = IsolationForest(n_estimators=100, contamination='auto', random_state=42)
    iso_forest.fit(p_vectors)
    if_scores = -iso_forest.decision_function(p_vectors)

    if np.sum(true_labels == 1) > 0:
        metrics = {'Reconstruction_MSE': mse_scores, 'Contrastive_Space_IsolationForest': if_scores}

        for metric_name, scores in metrics.items():
            plt.figure(figsize=(10, 6))
            sns.histplot(scores[true_labels == 0], color='blue', label='Background', kde=True, stat='density', alpha=0.5, bins=50)
            sns.histplot(scores[true_labels == 1], color='red', label='Anomalies', kde=True, stat='density', alpha=0.5, bins=50)
            plt.xlabel(f'{metric_name.replace("_", " ")} Score')
            plt.ylabel('Density')
            plt.title(f'{metric_name.replace("_", " ")} - {data_split.capitalize()}')
            plt.legend()
            plt.savefig(os.path.join(save_dir, f'loss_dist_{metric_name}_{data_split}.png'), bbox_inches='tight')
            plt.close()

        plt.figure(figsize=(10, 8))
        fpr_mse, tpr_mse, _ = roc_curve(true_labels, mse_scores)
        plt.plot(fpr_mse, tpr_mse, lw=2, label=f'Reconstruction (MSE) AUC = {auc(fpr_mse, tpr_mse):.3f}')
        fpr_if, tpr_if, _ = roc_curve(true_labels, if_scores)
        plt.plot(fpr_if, tpr_if, lw=2, label=f'Contrastive Space (IsoForest) AUC = {auc(fpr_if, tpr_if):.3f}')
        
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Hybrid SupCon ROC - {data_split.capitalize()}')
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(save_dir, f'roc_hybrid_comparison_{data_split}.png'), bbox_inches='tight')
        plt.close()
    else:
        print(f"Skipping ROC curve for {data_split.upper()} (No anomalies present).")

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
    
    encoder = HybridEncoder(latent_space_dim=args.latent_space_dim, proj_dim=64).to(device)
    decoder = Decoder(latent_space_dim=args.latent_space_dim).to(device)
    
    print(f"Loading model weights from: {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
        decoder.load_state_dict(checkpoint['decoder_state_dict'])
    else:
        encoder.load_state_dict(checkpoint)
        decoder.load_state_dict(checkpoint)

    evaluate_hybrid(valid_loader, encoder, decoder, device, args.save_dir, data_split="validation")
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