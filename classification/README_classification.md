# Jet Image Classification

This folder contains the supervised jet-image classification workflow for the project. The target is a 5-class jet tagging task with the following labels:

- gluon
- light quark
- top quark
- W boson
- Z boson

The codebase supports three related workflows:

- train a single backbone classifier with ResNet50 or InceptionV3
- train an ensemble classifier that combines both backbones
- tune and evaluate each setup with Optuna, TensorBoard, and Weights & Biases support

## Directory Overview

```text
classification/
├── README_classification.md
├── classifier_launcher.ipynb
├── dataset_download.ipynb
├── dataset_visual.ipynb
└── src/
    ├── evaluation.py
    ├── train.py
    ├── train_ensemble.py
    ├── tune.py
    ├── tune_ensemble.py
    ├── dataset/
    │   ├── __init__.py
    │   ├── dataloader.py
    │   └── dataset_preprocessing.py
    └── model/
        ├── __init__.py
        ├── ensemble.py
        ├── inceptionv3.py
        ├── mini_ensemble.py
        └── resnet.py
```

## Workflow

### 1. Prepare the dataset

Use `dataset_download.ipynb` to create or download the HDF5 dataset, and `dataset_visual.ipynb` to inspect the resulting jet images and label distribution.

The preprocessing script in `src/dataset/dataset_preprocessing.py` writes an HDF5 file with at least these datasets:

- `images`: jet images with shape `[N, 1, H, W]`
- `labels`: integer class labels in `[0, 4]`

The dataloader in `src/dataset/dataloader.py` then performs:

- stratified train/validation/test splitting
- mean and standard deviation computation on the training split
- resizing and normalization

### 2. Train a single backbone

Use `src/train.py` to train either:

- ResNet50
- InceptionV3

The script supports checkpointing, early stopping, TensorBoard logging, and WandB logging. Set `--mini` to use the smaller backbone variants defined in `src/model/mini_ensemble.py`.

### 3. Train the ensemble

Use `src/train_ensemble.py` to train the combined classifier built in `src/model/ensemble.py`.

The ensemble workflow is:

1. Load the ResNet and Inception feature extractors.
2. Concatenate their feature representations.
3. Train the final fully connected head first.
4. Optionally unfreeze the backbones for fine-tuning after warmup.

### 4. Tune hyperparameters

Use `src/tune.py` for single-backbone tuning and `src/tune_ensemble.py` for ensemble tuning.

Both scripts use Optuna and store the study state in SQLite so runs can be resumed.

### 5. Evaluate trained weights

Use `src/evaluation.py` to evaluate a saved checkpoint on validation and test splits. The script reports:

- loss and accuracy
- classification report
- confusion matrix
- multiclass ROC curves and AUC values

It also saves the plots to the selected output directory.

## Typical Run Order

1. Generate or download the HDF5 dataset.
2. Inspect the dataset visually.
3. Train a single backbone or the ensemble.
4. Tune hyperparameters if needed.
5. Evaluate the best checkpoint and save the plots.

## Common Output Files

Depending on the script you run, outputs are usually written to:

- model checkpoints in a `checkpoints/` directory
- TensorBoard logs inside `tensorboard_logs/`
- WandB run metadata
- Optuna SQLite studies and best-parameter text files
- evaluation plots such as confusion matrices and ROC curves

## Notes

- The default data file path in the scripts is a local HDF5 file such as `./dataset.h5` or a dataset path you provide explicitly.
- Most scripts expose `--img_size`, `--batch_size`, `--max_samples`, and `--mini` to make experiments cheaper during iteration.
- If you are running the full pipeline, keep the data preprocessing output, training checkpoints, and evaluation results in separate directories to avoid mixing generated artifacts with source files.
