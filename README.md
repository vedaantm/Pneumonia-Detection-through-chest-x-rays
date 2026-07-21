# Pneumonia-Detection-through-chest-x-rays
CNN-based pneumonia detection from chest X-rays using TensorFlow and Keras. Trained with Adam/SGD optimization, evaluated with confusion matrices and ROC curves (AUC: 0.96). Published at ICAIN 2024 — Springer LNNS Series, indexed in Scopus and Web of Science.
# Pneumonia Detection from Chest X-Rays

A deep learning image classification system that automatically 
detects pneumonia from chest X-ray images using Convolutional 
Neural Networks (CNN).

📄 **Published at ICAIN 2024 — Springer LNNS Series**  
🔗 Indexed in Scopus and Web of Science  
📖 [Read the paper](https://link.springer.com/chapter/10.1007/978-981-96-4319-6_33)

## Overview

Manual diagnosis of pneumonia from chest radiographs is 
time-consuming, inconsistent across radiologists, and often 
unavailable in resource-limited settings. This project builds 
an automated CNN-based classifier that distinguishes 
pneumonia-positive from normal chest X-rays with high accuracy 
— reducing diagnostic variability and supporting faster 
clinical decisions.

## Key Features

- **CNN Architecture** — Sequential model with convolutional 
  layers, ReLU activations, max-pooling, dropout regularization, 
  and a dense classification head
- **Dual Optimizer Comparison** — trained and evaluated with 
  both SGD and Adam; Adam achieved AUC of 0.96 on the ROC curve
- **Training Visualization** — accuracy and loss curves plotted 
  per epoch to monitor underfitting/overfitting
- **Evaluation** — confusion matrices, precision/recall, and ROC 
  curves on held-out test data
- **Model Persistence** — trained model saved in `.h5` format 
  for inference and reuse

## Results

| Optimizer | Normal (Correct) | Pneumonia (Correct) | AUC  |
|-----------|-----------------|---------------------|------|
| SGD       | 196 / 234       | 382 / 386           | —    |
| Adam      | 220 / 234       | 379 / 386           | 0.96 |

## Tech Stack

`Python` · `TensorFlow` · `Keras` · `Pandas` · `NumPy` · 
`Matplotlib` · `VisualKeras`
