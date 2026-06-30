# Jet Image Tagging and Anomaly Detection with Neural Networks

## Project Overview
This project explores Deep Learning applications in High Energy Physics (HEP), specifically focusing on **Jet Image Analysis**. The work is inspired by the paper *"Jet Image Tagging Using Deep Learning: An Ensemble Model"* by Bassa et al., and expands upon it to fit within a resource-constrained environment. 

The project is, at the moment, divided into two main tasks:
1. **Supervised Classification:** Building a computationally efficient ensemble of CNNs to classify different types of jets depending on the particle that originated them, aiming for high performance with reduced computational overhead.
2. **Unsupervised Anomaly Detection:** Developing a CNN-based Autoencoder to identify anomalous jets (potential signals of new physics) by training the network exclusively on known background data.

## Objectives
- **Data Engineering:** Extract particle cloud data using the [JetNet](https://github.com/jet-net/JetNet) library and convert the continuous $\eta-\phi$ spatial features into discrete 2D jet images (transverse momentum $p_T$ histograms).
- **Resource-Efficient Classification:** Implement an ensemble of "lightweight" CNNs to classify jets into categories (Top Quarks, Light Quarks, W and Z bosons) and evaluate them using standard HEP metrics (Accuracy, AUC-ROC).
- **New Physics Search:** Train an Autoencoder solely on QCD/Light quark background jets. Use the reconstruction error (MSE) as an anomaly score to isolate unseen signal jets.

## Structure
Our codebase is modularly designed to support two different architectures within the same pipeline:

```text
├── dataset_preprocessing.py  # JetNet point cloud to 2D image conversion & transforms
├── dataloader.py             # PyTorch Dataset and DataLoader definitions
├── models/
│   ├── ensemble_cnn.py       # Lightweight CNN components and Ensemble logic
│   └── autoencoder.py        # Encoder-Decoder architecture for anomaly detection
├── training.py               # Training loops (Supervised for Ensemble, Unsupervised for AE)
├── evaluation.py             # Metrics calculation (Accuracy, ROC curves, Reconstruction Loss distributions)
├── tune.py                   # Hyperparameter tuning script
├── main.py                   # Main execution script
└── README.md                 # Project documentation
