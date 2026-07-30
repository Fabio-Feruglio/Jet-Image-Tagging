import json
import os
import h5py 
import torch
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class JetImageAnomalyDataset(Dataset):
    def __init__(self, dataset_filepath, transform=None, indices=None, bg_classes=[0, 1]):
        self.filepath = dataset_filepath
        self.h5_file = None

        with h5py.File(self.filepath, "r") as f:
            labels_obj = f["labels"]
            total_length = int(labels_obj.shape[0])
        
        if indices is None:
            self.indices = np.arange(total_length)
        else:
            self.indices = indices
            
        self.transform = transform
        self.bg_classes = bg_classes

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        actual_idx = self.indices[idx]
        
        if self.h5_file is None:
            self.h5_file = h5py.File(self.filepath, 'r')
            
        image_np = self.h5_file['images'][actual_idx] 
        label_np = self.h5_file['labels'][actual_idx] 
        
        if self.bg_classes is not None:
            label_np = 0 if label_np in self.bg_classes else 1
        
        image_tensor = torch.from_numpy(image_np).to(dtype=torch.float32)
        label_tensor = torch.as_tensor(label_np, dtype=torch.long)

        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)
            
        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor
    
    def __del__(self):
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try: 
                self.h5_file.close()
            except Exception: 
                pass

def get_mean_and_std(dataloader, cache_file="dataset_stats.json"):
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            stats = json.load(f)
        return stats['mean'], stats['std']
    
    channels_sum = 0.0
    channels_sqrd_sum = 0.0
    num_pixels = 0.0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Calculating dataset stats"):
            channels_sum += images.sum()
            channels_sqrd_sum += (images ** 2).sum()
            num_pixels += images.numel()
            
    mean = channels_sum / num_pixels
    std = torch.sqrt((channels_sqrd_sum / num_pixels) - (mean ** 2))
    
    with open(cache_file, 'w') as f:
        json.dump({'mean': mean.item(), 'std': std.item()}, f)
        
    return mean.item(), std.item()

def get_dataloaders_tnt(data_filepath, bg_classes=[0, 1], img_size=128, batch_size=64, num_workers=0, 
                        num_train_samples=30000, threshold_percentile=90.0):
    
    with h5py.File(data_filepath, "r") as f:
        labels = np.asarray(f["labels"][:])

    all_indices = np.arange(labels.shape[0])
    bg_mask = np.isin(labels, bg_classes)
    
    bg_indices = all_indices[bg_mask]
    anomaly_indices = all_indices[~bg_mask]
    
    # Shuffle iniziale per garantire la casualità
    np.random.seed(42)
    np.random.shuffle(bg_indices)
    np.random.shuffle(anomaly_indices)

    # --- 1. CALCOLI MATEMATICI SICURI E DINAMICI ---
    ae1_size = num_train_samples
    
    if ae1_size >= len(bg_indices):
        raise ValueError(f"Errore: Hai chiesto {ae1_size} campioni per AE1, ma hai solo {len(bg_indices)} campioni di background totali.")

    bg_remaining = len(bg_indices) - ae1_size
    anom_remaining = len(anomaly_indices)

    # Riserviamo campioni per l'Evaluation Set (Max 5000 per classe, o il 10% di quello che resta)
    eval_half = min(5000, bg_remaining // 10, anom_remaining // 10)
    
    if eval_half == 0:
         raise ValueError("Non ci sono abbastanza dati per creare un set di valutazione!")

    # Calcoliamo quanto servirebbe idealmente per il Tagging Set
    fraction_kept = (100.0 - threshold_percentile) / 100.0
    ideal_tagging_half = int((num_train_samples / fraction_kept) / 2)
    
    # Prendiamo il numero ideale, MA limitato da quanti dati abbiamo REALMENTE a disposizione
    tagging_half = min(ideal_tagging_half, bg_remaining - eval_half, anom_remaining - eval_half)

    # --- 2. AFFETTAMENTO (SLICING) DISGIUNTO ---
    idx_bkg = 0
    ae1_bg_idx = bg_indices[idx_bkg : idx_bkg + ae1_size]
    idx_bkg += ae1_size
    tagging_bg_idx = bg_indices[idx_bkg : idx_bkg + tagging_half]
    idx_bkg += tagging_half
    eval_bg_idx = bg_indices[idx_bkg : idx_bkg + eval_half]

    idx_anom = 0
    tagging_anom_idx = anomaly_indices[idx_anom : idx_anom + tagging_half]
    idx_anom += tagging_half
    eval_anom_idx = anomaly_indices[idx_anom : idx_anom + eval_half]

    # Mix and Shuffle per i set misti
    tag_idx = np.concatenate([tagging_bg_idx, tagging_anom_idx])
    eval_idx = np.concatenate([eval_bg_idx, eval_anom_idx])
    np.random.shuffle(tag_idx)
    np.random.shuffle(eval_idx)

    # Split interno 85/15 per AE1
    ae1_train, ae1_val = train_test_split(ae1_bg_idx, test_size=0.15, random_state=42)

    print(f"\n--- Splitting Dataset TNT Disaccoppiato ---")
    print(f"AE1 Train+Val : {len(ae1_train) + len(ae1_val)} campioni (100% BKG)")
    print(f"Tagging Set   : {len(tag_idx)} campioni (50/50 BKG-Anom)")
    print(f"-> Con un taglio al {threshold_percentile}°, AE2 otterrà esattamente {int(len(tag_idx)*fraction_kept)} campioni!")
    print(f"Eval Set      : {len(eval_idx)} campioni (50/50 BKG-Anom)")
    print(f"-------------------------------------------")

    # Transforms
    stat_loader = DataLoader(JetImageAnomalyDataset(data_filepath, indices=ae1_train, bg_classes=bg_classes), batch_size=512)
    calc_mean, calc_std = get_mean_and_std(stat_loader)
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(mean=[calc_mean], std=[calc_std]),
    ])

    ds_train = JetImageAnomalyDataset(data_filepath, transform=transform, indices=ae1_train, bg_classes=bg_classes)
    ds_val = JetImageAnomalyDataset(data_filepath, transform=transform, indices=ae1_val, bg_classes=bg_classes)
    ds_tag = JetImageAnomalyDataset(data_filepath, transform=transform, indices=tag_idx, bg_classes=bg_classes)
    ds_eval = JetImageAnomalyDataset(data_filepath, transform=transform, indices=eval_idx, bg_classes=bg_classes)

    pin_mem = torch.cuda.is_available()
    loaders = [
        DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_mem),
        DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem),
        DataLoader(ds_tag, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem),
        DataLoader(ds_eval, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_mem)
    ]
    
    return loaders[0], loaders[1], loaders[2], loaders[3]