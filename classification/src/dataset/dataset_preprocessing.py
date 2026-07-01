# Import and preprocess data from jetnet package
import torch
import h5py
import numpy as np
from tqdm import tqdm
from jetnet.datasets import JetNet
from jetnet.utils import to_image
import os
import argparse

def process_and_save_hdf5(output_filepath='jet_images_299.h5', 
                          data_dir='./JetNet_dataset', 
                          jet_types=['t', 'q', 'g', 'w', 'z'], 
                          im_size=299, 
                          maxR=0.4, 
                          batch_size=1000):
    """
    Download JetNet dataset, convert to images and save in .hdf5 file
    """

    os.makedirs(data_dir, exist_ok=True)
    
    # Download the dataset using the JetNet funtion
    data_args = {
        "jet_type": jet_types,
        "data_dir": data_dir,
        "particle_features": "all",
        "num_particles": 30,
        "jet_features": "all",
        "download": True
    }

    print(f"Loading dataset JetNet for types: {jet_types}...")
    particle_data, jet_data = JetNet.getData(**data_args)
    total_samples = len(particle_data)
    print(f"Found {total_samples} total samples.")

    # Initialize hdf5 files
    print(f"Initializing output file: {output_filepath}")
    with h5py.File(output_filepath, 'w') as h5f:

        # Shape: (Batch, Channel, Height, Width) -> (N, 1, 299, 299)
        dset_images = h5f.create_dataset(
            'images', 
            shape=(0, 1, im_size, im_size), 
            maxshape=(None, 1, im_size, im_size), 
            dtype='float32',
            batchs=(batch_size, 1, im_size, im_size) # batch reading
        )
        
        dset_labels = h5f.create_dataset(
            'labels', 
            shape=(0,), 
            maxshape=(None,), 
            dtype='int64'
        )

        # Conversion to images with JetNet native function
        print("3. Generazione immagini e salvataggio a blocchi...")
        for start_idx in tqdm(range(0, total_samples, batch_size)):
            end_idx = min(start_idx + batch_size, total_samples)
            
            # Define the batch
            batch_particles = particle_data[start_idx:end_idx]
            batch_labels = jet_data[start_idx:end_idx, 0]
            
            # Convert to images
            batch_images = [to_image(p, im_size=im_size, maxR=maxR) for p in batch_particles]
            
            # Convert to numpy array
            batch_images_np = np.expand_dims(np.array(batch_images, dtype=np.float32), axis=1)
            
            # Resize hdf5 dataset to include new data
            dset_images.resize((dset_images.shape[0] + batch_images_np.shape[0]), axis=0)
            dset_labels.resize((dset_labels.shape[0] + batch_labels.shape[0]), axis=0)
            
            # Write the new batch on the file
            dset_images[start_idx:end_idx] = batch_images_np
            dset_labels[start_idx:end_idx] = batch_labels

    print(f"\nPre-processing complete: dataset saved in {output_filepath}")

def main(args):
    process_and_save_hdf5(args.output_file, args.data_dir, ['t', 'q', 'g', 'w', 'z'], args.im_size, args.maxR, args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and pre-process the JetNet dataset")
    parser.add_argument('--output_file', type=str, default="jet_images_299.h5", help='Output file path name')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch dimension (not too large)')
    parser.add_argument('--maxR', type=float, default=0.4, help='Max range of pseudorapidity / azimuthal angle')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory for raw data')
    parser.add_argument('--im_size', type=int, default=299, help='Size of image in pixels')
    args = parser.parse_args()
    main(args)