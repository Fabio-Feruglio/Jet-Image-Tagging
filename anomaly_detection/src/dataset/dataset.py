import h5py
import torch
from torch.utils.data import Dataset
import numpy as np

class JetH5Dataset(Dataset):
    def __init__(self, filepath, mode='train'):
        """
        Args:
            filepath (str): Path al file .h5
            mode (str): 'train' (solo quark e gluoni) o 'test' (tutto il dataset)
        """
        super().__init__()
        self.filepath = filepath
        self.mode = mode
        
        
        with h5py.File(self.filepath, 'r') as f:
            labels = f['labels'][:]
            
        # JET_TYPES = ["g", "q", "t", "w", "z"]
        # Assume that g=0, q=1 (Background) e t=2, w=3, z=4 (Anomalies)
        if mode == 'train':
            # Let's take only Gluons (0) and Quarks (1)
            mask = (labels == 0) | (labels == 1)
            self.indices = np.where(mask)[0]
        else:
            # In test mode, we take all samples
            self.indices = np.arange(len(labels))
            
        self.file = None
        self.images = None
        self.labels = None

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        
        if self.file is None:
            self.file = h5py.File(self.filepath, 'r')
            self.images = self.file['images']
            self.labels = self.file['labels']
            
        real_idx = self.indices[idx]
        
        # Read from hdf5
        image = self.images[real_idx]
        label = self.labels[real_idx]
        
        # Convert in tensor
        image_tensor = torch.tensor(image, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return image_tensor, label_tensor