import h5py 
import torch
import numpy as np
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from data_augmentation import RandomDeadPixel

### Dataset Classes and Dataloader functions

class JetImageDataset(Dataset):
    def __init__(self, dataset_filepath, transform = None, indices = None):
        """
        PyTorch dataset for loading 2D jet images from an HDF5 file
        """
        self.filepath = dataset_filepath
        self.h5_file = None

        with h5py.File(self.filepath, 'r') as f:
            """
            labels_obj = f["labels"]

            if not isinstance(labels_obj, h5py.Dataset):
                raise TypeError("Expected 'labels' to be an HDF5 dataset")

            if labels_obj.shape is None or len(labels_obj.shape) == 0:
                raise ValueError("'labels' must be at least 1D")

            total_length = int(labels_obj.shape[0])
            """
            total_length = len(f['labels']) # if this does not work, uncomment code before...
        
        if self.indices is None:
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
        image_tensor = torch.tensor(image_np, dtype=torch.float32)
        label_tensor = torch.tensor(label_np, dtype=torch.long)

        if self.transform:
            image_tensor = self.transform(image_tensor)
            
        return image_tensor, label_tensor


def get_mean_and_std(dataloader):
    """
    Compute mean and standard deviation of the dataset for normalization.
    """
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
    
    return mean.item(), std.item()



def get_dataloaders(data_filepath = "./dataset.h5", img_size = 299, batch_size = 64, num_workers = 0):
    """
    Prepare the dataloaders for training, test and validation
    """

    # Split the dataset into training, validation, and test sets
    with h5py.File(data_filepath, 'r') as f:
        total_samples = len(f['labels'])
    all_indices = np.arange(total_samples)
    train_idx, temp_idx = train_test_split(all_indices, test_size=0.2, random_state=42)
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.5, random_state=42)

    # Compute mean and std for normalization
    raw_train_dataset = JetImageDataset(dataset_filepath = data_filepath, indices = train_idx)

    stat_loader = DataLoader(dataset = raw_train_dataset, batch_size = 256, shuffle = False, num_workers = 2)
    calculated_mean, calculated_std = get_mean_and_std(stat_loader)

    # Transforms for data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)), # Resize
        transforms.Normalize(mean = calculated_mean, std = calculated_std), # Normalize with calculated stats
        #transforms.RandomErasing(),
        #RandomDeadPixel(p = 0.1, max_dead_pixels = 10) # Custom transform to simulate dead pixels
    ])
    
    # For the evaluation we do not augment data
    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Normalize(mean = calculated_mean, std = calculated_std), # Normalize with calculated stats
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