# Import and pre-processing from JetNet dataset
import h5py
import numpy as np
from tqdm import tqdm
from jetnet.datasets import JetNet
from jetnet.utils import to_image
import os
import argparse


def process_and_save_hdf5(output_filepath = 'jet_images_299.h5',
                           data_dir = './JetNet_dataset',
                           jet_types = ['t', 'q', 'g', 'w', 'z'],
                           im_size = 299,
                           maxR = 0.4,
                           batch_size = 1000,
                           compression = 'gzip',
                           compression_opts = 4,
                           image_dtype = 'float32'):
    
    os.makedirs(data_dir, exist_ok = True)
    
    # We just need (eta, phi, pt) and the mask for each particle for image conversion
    # Also we don't need all jet features for our task
    particle_feats = ["etarel", "phirel", "ptrel", "mask"]
    jet_feats = ["type", "pt", "eta", "mass", "num_particles"]

    data_args = {
        "jet_type": jet_types,                  # ['t', 'q', 'g', 'w', 'z'] or a subset
        "data_dir": data_dir,                   # directory where the raw JetNet data is stored
        "particle_features": particle_feats,    # features to load for each particle
        "num_particles": 30,                    # maximum number of particles per jet (pad with zeros if fewer)
        "jet_features": jet_feats,              # features to load for each jet
        "download": True,                       # download the dataset, if not already present in data_dir
    }

    print(f"Loading dataset JetNet for types: {jet_types}...")

    particle_data, jet_data = JetNet.getData(**data_args)
    total_samples = len(particle_data)

    print(f"Found {total_samples} total samples.")

    print(f"Initializing output file: {output_filepath}")
    with h5py.File(output_filepath, 'w') as h5f:
        
        dset_images = h5f.create_dataset(
            'images',
            shape = (0, 1, im_size, im_size),
            maxshape = (None, 1, im_size, im_size), # allow unlimited growth in the first dimension
            dtype = image_dtype,
            chunks = (1, 1, im_size, im_size), # chunk size for efficient writing
            compression = compression,
            compression_opts = compression_opts if compression == 'gzip' else None,
            shuffle = True,
        )

        dset_labels = h5f.create_dataset(
            'labels',
            shape = (0, ),
            maxshape = (None, ),
            dtype = 'int64',
        )

        print("Generating images from particle data...")

        for start_idx in tqdm(range(0, total_samples, batch_size), desc="Processing batches", unit="batch"):
            end_idx = min(start_idx + batch_size, total_samples)

            batch_particles = particle_data[start_idx:end_idx]  # [batch_size, max_particles, part_features] = [B, 30, 4]
            batch_labels = jet_data[start_idx:end_idx, 0]       # [batch_size] = [B]

            # Separate first 3 columns = (eta, phi, pt); 
            # fourth column = binary mask (1 = real particle, 0 = padding).
            coords = batch_particles[..., :3].astype(np.float32)
            masks = batch_particles[..., 3]
            
            # Use native JetNet function to convert particle coordinates to images
            # to_image can accept a batch of jets and return a batch of images
            batch_images_np = to_image(coords, im_size=im_size, maxR=maxR, mask=masks)
            batch_images_np = np.expand_dims(
                batch_images_np.astype(image_dtype), axis=1
            )  # convert [B, im_size, im_size] to [B, 1, im_size, im_size]

            # Save the new batch to the hdf5 dataset
            n_new = batch_images_np.shape[0]
            current_size = dset_images.shape[0]
            dset_images.resize(current_size + n_new, axis=0)
            dset_labels.resize(current_size + n_new, axis=0)

            dset_images[current_size:current_size + n_new] = batch_images_np
            dset_labels[current_size:current_size + n_new] = batch_labels

    size_gb = os.path.getsize(output_filepath) / (1024 ** 3)
    print(f"\nPre-processing complete: dataset saved in {output_filepath}")
    print(f"Final size of the file: {size_gb:.2f} GB")


def main(args):
    process_and_save_hdf5(
        args.output_file,
        args.data_dir,
        ['t', 'q', 'g', 'w', 'z'],
        args.im_size,
        args.maxR,
        args.batch_size,
        compression=args.compression,
        compression_opts=args.compression_opts,
        image_dtype=args.dtype,
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load and pre-process the JetNet dataset")
    parser.add_argument('--output_file', type=str, default="jet_images_299.h5", help='Output file path name')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch dimension (not too large)')
    parser.add_argument('--maxR', type=float, default=0.4, help='Max range of pseudorapidity / azimuthal angle')
    parser.add_argument('--data_dir', type=str, default='./dataset', help='Directory for raw data')
    parser.add_argument('--im_size', type=int, default=299, help='Size of image in pixels')
    parser.add_argument('--compression', type=str, default='gzip', choices=['gzip', 'lzf'],
                         help="HDF5 compression filter. 'lzf' is faster, 'gzip' compresses more")
    parser.add_argument('--compression_opts', type=int, default=4,
                         help='Gzip compression level (0-9, ignored if --compression lzf)')
    parser.add_argument('--dtype', type=str, default='float32', choices=['float32', 'float16'],
                         help='Precision with which to save image pixels (float16 halves the space)')
    args = parser.parse_args()
    main(args)
