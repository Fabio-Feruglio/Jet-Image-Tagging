# Anomaly detection task

## What are we doing?

Dataset: 
- Training & val: jet images from qcd background (original labels: 'q' and 'g')
-  Anomalies: use jets from 't' and 'z' or 'w'

Objective: train a sort of (variational) autoencoder to reconstruct very well jet images from gluons and light quarks, then check if we can set up a reconstruction score that allows us to find anomalies at test time, i.e. jets generated from other heavy particles.

## Some things to take into account

- Probably we want to reduce the dimensionality of the images (that are highly sparse): build models that can be adapted to take images that are 128x128 or even 64x64
- We cannot use the same exact model from the classification task, as it was trained on all classes
- Add here...
  
## Proposed architectures

- write here...