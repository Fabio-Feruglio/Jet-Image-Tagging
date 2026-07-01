import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt 
import numpy as np 
import random

def get_dataloaders(data_dir='dataset', batch_size=256):
    """
    Downloads the dataset FashionMNIST and returns DataLoader for train and test
    """
    # Define transformations
    train_transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Download the dataset
    train_dataset = torchvision.datasets.FashionMNIST(data_dir, train=True, download=True, transform=train_transform)
    test_dataset  = torchvision.datasets.FashionMNIST(data_dir, train=False, download=True, transform=test_transform)

    # Create DataLoader
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    """
    ### Plot some sample
    label_names=['t-shirt','trouser','pullover','dress','coat','sandal','shirt',
                  'sneaker','bag','boot']
    fig, axs = plt.subplots(5, 5, figsize=(8,8))
    for ax in axs.flatten():
        # random.choice allows to randomly sample from a list-like object (basically anything that can be accessed with an index, like our dataset)
        img, label = random.choice(train_dataset)
        ax.imshow(np.array(img), cmap='gist_gray')
        ax.set_title(f'Label: {label_names[label]} [{label}]')
        ax.set_xticks([])
        ax.set_yticks([])
    plt.tight_layout()
    """

    return train_dataloader, test_dataloader, test_dataset