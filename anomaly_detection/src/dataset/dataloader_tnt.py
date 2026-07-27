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
            if not isinstance(labels_obj, h5py.Dataset):
                raise TypeError("Expected 'labels' to be an HDF5 dataset")
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
        print(f"Loading cached mean and std from {cache_file}...")
        with open(cache_file, 'r') as f:
            stats = json.load(f)
        return stats['mean'], stats['std']
    
    channels_sum = 0.0
    channels_sqrd_sum = 0.0
    num_pixels = 0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader, desc="Calculating dataset stats"):
            channels_sum += images.sum()
            channels_sqrd_sum += (images ** 2).sum()
            num_pixels += images.numel()
            
    mean = channels_sum / num_pixels
    variance = (channels_sqrd_sum / num_pixels) - (mean ** 2)
    std = torch.sqrt(variance)

    with open(cache_file, 'w') as f:
        json.dump({'mean': mean.item(), 'std': std.item()}, f)
    
    return mean.item(), std.item()


def get_dataloaders_tnt(data_filepath="./dataset.h5", bg_classes=[0, 1], img_size=128, batch_size=64, num_workers=0, max_samples=50000):
    with h5py.File(data_filepath, "r") as f:
        labels = np.asarray(f["labels"][:])

    all_indices = np.arange(labels.shape[0])
    bg_mask = np.isin(labels, bg_classes)
    anomaly_mask = ~bg_mask

    bg_indices = all_indices[bg_mask]
    anomaly_indices = all_indices[anomaly_mask]
    
    bg_labels = labels[bg_mask]
    anomaly_labels = labels[anomaly_mask]

    if max_samples is not None and max_samples < len(bg_indices):
        bg_indices, _, bg_labels, _ = train_test_split(
            bg_indices, bg_labels, train_size=max_samples, random_state=42, stratify=bg_labels
        )

    # 1. Divisione Background: 60% Train+Val AE1, 20% Tagging, 20% Eval
    bg_train_val, bg_tag_eval, labels_train_val, _ = train_test_split(
        bg_indices, bg_labels, test_size=0.40, random_state=42, stratify=bg_labels
    )
    bg_train, bg_val = train_test_split(bg_train_val, test_size=1/6, random_state=42)
    bg_tag, bg_eval = train_test_split(bg_tag_eval, test_size=0.50, random_state=42)

    # 2. Campionamento delle Anomalie (FIX: Pareggiamo il numero col background)
    num_anom_needed = len(bg_tag_eval)
    num_anom_needed = min(num_anom_needed, len(anomaly_indices))

    sampled_anomaly_idx, _, sampled_anomaly_labels, _ = train_test_split(
        anomaly_indices, anomaly_labels,
        train_size=num_anom_needed,
        random_state=42,
        stratify=anomaly_labels
    )

    # Dividiamo le anomalie campionate 50% al Tagging e 50% all'Eval
    anom_tag, anom_eval = train_test_split(
        sampled_anomaly_idx, test_size=0.50, random_state=42, stratify=sampled_anomaly_labels
    )

    # Unione per i set misti
    tag_idx = np.concatenate([bg_tag, anom_tag])
    eval_idx = np.concatenate([bg_eval, anom_eval])
    np.random.shuffle(tag_idx)
    np.random.shuffle(eval_idx)

    print(f"\n--- Splitting Dataset TNT ---")
    print(f"AE1 Train (solo BG): {len(bg_train)} | Val (solo BG): {len(bg_val)}")
    print(f"Tagging Set (misto): {len(tag_idx)} | Eval Set (misto): {len(eval_idx)}")

    raw_train_dataset = JetImageAnomalyDataset(dataset_filepath=data_filepath, indices=bg_train, bg_classes=bg_classes)
    stat_loader = DataLoader(raw_train_dataset, batch_size=512, shuffle=False)
    calc_mean, calc_std = get_mean_and_std(stat_loader)

    transform = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias=True),
        transforms.Normalize(mean=[calc_mean], std=[calc_std]),
    ])

    ds_train = JetImageAnomalyDataset(data_filepath, transform=transform, indices=bg_train, bg_classes=bg_classes)
    ds_val = JetImageAnomalyDataset(data_filepath, transform=transform, indices=bg_val, bg_classes=bg_classes)
    ds_tag = JetImageAnomalyDataset(data_filepath, transform=transform, indices=tag_idx, bg_classes=bg_classes)
    ds_eval = JetImageAnomalyDataset(data_filepath, transform=transform, indices=eval_idx, bg_classes=bg_classes)

    pin_mem = torch.cuda.is_available()
    loaders = [DataLoader(ds, batch_size=batch_size, shuffle=(i==0), num_workers=num_workers, pin_memory=pin_mem) 
               for i, ds in enumerate([ds_train, ds_val, ds_tag, ds_eval])]
    
    return loaders[0], loaders[1], loaders[2], loaders[3]