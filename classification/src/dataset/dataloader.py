import json
import os

import h5py 
import torch
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
try:
    from .data_augmentation import RandomDeadPixel
except ImportError:  # pragma: no cover
    from data_augmentation import RandomDeadPixel

### Dataset Classes and Dataloader functions

class JetImageDataset(Dataset):
    def __init__(self, dataset_filepath, transform = None, indices = None):
        """
        PyTorch dataset for loading 2D jet images from an HDF5 file
        """
        self.filepath = dataset_filepath
        self.h5_file = None

        with h5py.File(self.filepath, "r") as f:
            labels_obj = f["labels"]

            if not isinstance(labels_obj, h5py.Dataset):
                raise TypeError("Expected 'labels' to be an HDF5 dataset")

            if labels_obj.shape is None or len(labels_obj.shape) == 0:
                raise ValueError("'labels' must be at least 1D")

            total_length = int(labels_obj.shape[0])
        
        if indices is None:
            self.indices = np.arange(total_length)
        else:
            self.indices = indices

        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):

        # Map the index to the actual index in the h5 file
        actual_idx = self.indices[idx]
        
        # Open the h5 file if it is not already open
        if self.h5_file is None:
            self.h5_file = h5py.File(self.filepath, 'r')
            
        # Read image and label from the HDF5 file
        image_np = self.h5_file['images'][actual_idx]
        label_np = self.h5_file['labels'][actual_idx]
        
        # Convert to torch tensors
        image_tensor = torch.from_numpy(image_np).to(dtype=torch.float32)
        label_tensor = torch.as_tensor(label_np, dtype=torch.long)

        if image_tensor.ndim == 2:
            image_tensor = image_tensor.unsqueeze(0)

        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor
    
    def __del__(self):
        """
        Destructor
        """
        if hasattr(self, 'h5_file') and self.h5_file is not None:
            try:
                self.h5_file.close()
            except Exception:
                pass


def get_mean_and_std(dataloader, cache_file="dataset_stats.json"):
    """
    Compute mean and standard deviation of the dataset for normalization.
    """
    if os.path.exists(cache_file):
        print(f"Loading cached mean and std from {cache_file}...")
        with open(cache_file, 'r') as f:
            stats = json.load(f)
        return stats['mean'], stats['std']
    
    channels_sum = 0.0
    channels_sqrd_sum = 0.0
    num_pixels = 0
    
    with torch.no_grad():
        for images, _ in tqdm(dataloader):

            channels_sum += images.sum()
            channels_sqrd_sum += (images ** 2).sum()
            num_pixels += images.numel()
            
    mean = channels_sum / num_pixels
    variance = (channels_sqrd_sum / num_pixels) - (mean ** 2)
    std = torch.sqrt(variance)

    with open(cache_file, 'w') as f:
        json.dump({'mean': mean.item(), 'std': std.item()}, f)
    
    return mean.item(), std.item()



def get_dataloaders(data_filepath = "./dataset.h5", img_size = 299, batch_size = 64, num_workers = 0, max_samples = None):
    """
    Prepare the dataloaders for training, test and validation
    """

    # Split the dataset into training, validation, and test sets (stratified by label)
    with h5py.File(data_filepath, "r") as f:
        labels_obj = f["labels"]
        if not isinstance(labels_obj, h5py.Dataset):
            raise TypeError("'labels' must be an HDF5 Dataset")
        labels = np.asarray(labels_obj[:])

    all_indices = np.arange(labels.shape[0])

    if max_samples is not None and max_samples < len(all_indices):
        print(f"Subsampling dataset to {max_samples} samples (stratified)...")
        all_indices, _, labels, _ = train_test_split(
            all_indices, labels, train_size=max_samples, random_state=42, stratify=labels
        )
    
    train_idx, temp_idx, train_labels, temp_labels = train_test_split(
        all_indices, labels, test_size=0.2, random_state=42, stratify=labels
    )
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.5, random_state=42, stratify=temp_labels 
    )

    # Compute mean and std for normalization
    raw_train_dataset = JetImageDataset(dataset_filepath = data_filepath, indices = train_idx)

    stat_loader = DataLoader(dataset = raw_train_dataset, batch_size = 512, shuffle = False, num_workers = 0)

    calculated_mean, calculated_std = get_mean_and_std(stat_loader)

    # Transforms for data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias = True), # Resize
        transforms.Normalize(mean = [calculated_mean], std = [calculated_std]), # Normalize with calculated stats
        #transforms.RandomErasing(),
        #RandomDeadPixel(p = 0.1, max_dead_pixels = 10) # Custom transform to simulate dead pixels
    ])
    
    # For the evaluation we do not augment data
    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size), antialias = True),
        transforms.Normalize(mean = [calculated_mean], std = [calculated_std]), # Normalize with calculated stats
    ])


    # Datasets creation
    train_dataset = JetImageDataset(dataset_filepath = data_filepath, 
                                    transform = train_transforms, 
                                    indices = train_idx)
    valid_dataset = JetImageDataset(dataset_filepath = data_filepath, 
                                    transform = eval_transforms, 
                                    indices = val_idx)
    test_dataset  = JetImageDataset(dataset_filepath = data_filepath, 
                                    transform = eval_transforms, 
                                    indices = test_idx)

    # DataLoaders creation
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(dataset = train_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = True,
                                  num_workers = num_workers, 
                                  pin_memory = pin_memory)
    valid_dataloader = DataLoader(dataset = valid_dataset, 
                                  batch_size = batch_size, 
                                  shuffle = False,
                                  num_workers = num_workers, 
                                  pin_memory = pin_memory)
    test_dataloader = DataLoader(dataset = test_dataset, 
                                 batch_size = batch_size, 
                                 shuffle = False,
                                 num_workers = num_workers, 
                                 pin_memory = pin_memory)

    return train_dataloader, valid_dataloader, test_dataloader