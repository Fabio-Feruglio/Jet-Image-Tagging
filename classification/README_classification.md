# Jet Image Classification with an Ensemble Model

This folder contains the supervised classification part of the project: a 5-class jet image classifier built around an ensemble of two CNN backbones, **ResNet50** and **InceptionV4**.

The goal is to classify jet images into the following classes:

- gluon
- light quark
- top quark
- W boson
- Z boson

The code supports training individual backbones, training the ensemble, hyperparameter tuning, and model evaluation with classification metrics and plots.

## What is included

- PyTorch dataset and dataloader utilities for jet images stored in HDF5 format
- Data normalization and train/validation/test splitting with stratification
- ResNet50 and InceptionV4 models
- An ensemble model that concatenates the two backbone feature representations and learns a final classifier head
- Training scripts with checkpointing, early stopping, TensorBoard logging, and Weights & Biases logging
- Evaluation script with accuracy, classification report, confusion matrix, and multiclass ROC curves
- Optuna-based hyperparameter tuning utilities
- Jupyter notebooks for dataset inspection, downloading, tuning, and training

## Repository structure

```text
classification/
├── README.md
├── classifier_train_launcher.ipynb
├── classifier_tuner.ipynb
├── dataset_download.ipynb
├── dataset_visual.ipynb
└── src/
    ├── evaluation.py
    ├── train.py
    ├── train_ensemble.py
    ├── tune.py
    ├── tune_old.py
    ├── dataset/
    │   ├── __init__.py
    │   ├── data_augmentation.py
    │   ├── dataloader.py
    │   └── dataset_preprocessing.py
    └── model/
        ├── __init__.py
        ├── ensemble.py
        ├── inception.py
        └── resnet.py
```

## Data format

The dataloader expects an HDF5 file containing at least two datasets:

- `images`: jet images
- `labels`: class labels

The loader splits the data into training, validation, and test sets using a stratified split. It also computes the dataset mean and standard deviation on the training split and uses them for normalization.

## Model overview

### ResNet50 and InceptionV4

The backbone models are standard CNN classifiers adapted to the 5-class jet tagging task.

### Ensemble model

The ensemble is defined in `src/model/ensemble.py` and works as follows:

1. Load the two backbones.
2. Remove their final classification layers.
3. Extract feature vectors from both networks.
4. Concatenate the features.
5. Pass the merged representation through a small fully connected head.

This lets the model combine complementary information from the two architectures.
