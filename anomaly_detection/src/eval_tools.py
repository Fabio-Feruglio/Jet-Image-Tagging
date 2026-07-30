import argparse
import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


def save_reconstruction_pairs_by_class(original, reconstructed, labels, save_dir, data_split, model_name="", num_per_class=3):
    
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
        print(f"No images found for {data_split}.")
        return

    fig, axes = plt.subplots(nrows=total_images, ncols=2, figsize=(8, 3 * total_images))
    
    if total_images == 1:
        axes = [axes]
        
    for i, idx in enumerate(selected_indices):
        label_type = "Background (Normal)" if labels_np[idx] == 0 else "Anomaly (New Physics)"
        
        img_in = orig_np[idx].squeeze()
        img_out = recon_np[idx].squeeze()

        ax_orig = axes[i][0]
        ax_orig.imshow(img_in, cmap='gray', vmin=0.0, vmax=1.0)
        ax_orig.set_title(f"Input - {label_type}")
        ax_orig.axis('off')
        
        ax_recon = axes[i][1]
        ax_recon.imshow(img_out, cmap='gray', vmin=0.0, vmax=1.0)
        ax_recon.set_title(f"VAE Recon - {label_type}")
        ax_recon.axis('off')
        
    plt.tight_layout()
    
    save_path = os.path.join(img_dir, f"reconstructions_comparison_{data_split}.png")
    plt.savefig(save_path, bbox_inches='tight')
    plt.close(fig)
    print(f"Reconstructions saved to: {save_path}")
