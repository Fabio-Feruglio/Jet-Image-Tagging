import os
import gzip
import shutil
import h5py # type: ignore
import torch
from pathlib import Path
import gdown  # type: ignore
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

class ImageDataset(Dataset):
    def __init__(self, dataset_folder, dataset_type, transform=None, max_samples=None):
        x_path = os.path.join(dataset_folder, f'camelyonpatch_level_2_split_{dataset_type}_x.h5')
        y_path = os.path.join(dataset_folder, f'camelyonpatch_level_2_split_{dataset_type}_y.h5')
        
        self.x = h5py.File(x_path, 'r')['x']
        self.y = h5py.File(y_path, 'r')['y']
        
        self.transform = transform
        self.max_samples = max_samples

    def __len__(self):
        if self.max_samples is None:
            return len(self.x)
        return min(self.max_samples, len(self.x))

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx, 0, 0].astype(float)
        
        if self.transform:
            x = self.transform(x)
            
        return x, y

def prepare_data(data_dir: str):
    """
    Check if h5 files exist. Otherwise download and extract them
    """
    DATA_DIR = Path(data_dir)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    required_files = [
        "camelyonpatch_level_2_split_train_x.h5",
        "camelyonpatch_level_2_split_train_y.h5",
        "camelyonpatch_level_2_split_valid_x.h5",
        "camelyonpatch_level_2_split_valid_y.h5",
        "camelyonpatch_level_2_split_test_x.h5",
        "camelyonpatch_level_2_split_test_y.h5",
    ]

    # Check if the files are missing
    if not all((DATA_DIR / f).exists() for f in required_files):
        print(f"[{data_dir}] Starting download...")
        gdown.download_folder(
            url="https://drive.google.com/drive/folders/1VITWmHRhh3cD1ZR9VjQOdPY8aUvfmam6?usp=drive_link",
            output=str(DATA_DIR),
            quiet=False,
        )
    # Get the .gz if available
    gz_files = list(DATA_DIR.glob("*.h5.gz"))
    if gz_files:
        print(f"[{data_dir}] Extracting compressed files...")
        for gz_file in gz_files:
            output_file = gz_file.with_suffix("")
            with gzip.open(gz_file, "rb") as f_in, open(output_file, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            gz_file.unlink() 

    print(f"[{data_dir}] All files are ready!")

def get_dataloaders(data_dir="./data_lab04", batch_size=64, max_train_samples=32000):
    """
    Prepare the dataloaders for training, test and validation
    """
    
    prepare_data(data_dir)

    # Transforms for data augmentation
    train_transforms = transforms.Compose([
        transforms.ToTensor(), 
        transforms.RandomAffine((0.05, 0.05)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip()
    ])
    
    # For the evaluation we do not augment data
    eval_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    # Datasets creation
    train_dataset = ImageDataset(data_dir, "train", transform=train_transforms, max_samples=max_train_samples)
    valid_dataset = ImageDataset(data_dir, "valid", transform=eval_transforms)
    test_dataset  = ImageDataset(data_dir, "test", transform=eval_transforms)

    # DataLoaders creation
    num_workers = 0
    pin_memory = torch.cuda.is_available()

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=pin_memory)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                  num_workers=num_workers, pin_memory=pin_memory)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                 num_workers=num_workers, pin_memory=pin_memory)

    return train_dataloader, valid_dataloader, test_dataloader