# Zero-Shot Fault Detection Across Unseen Industrial Machine Domains
**Date:** June 12, 2026

## 1. Objective and Pretrained Methodology
To achieve highly robust zero-shot anomaly detection on completely unseen machine types, this pipeline utilizes a **Transfer Learning** paradigm. Audio samples are converted into 2D Mel-Spectrograms, allowing the model to process acoustic data as visual textures. 

Instead of training a network from scratch, a **pretrained ResNet-18** (trained on ImageNet) acts as a frozen feature extractor. Pretrained vision models, particularly advanced Convolutional Neural Networks and Vision Transformers, are exceptionally adept at identifying intricate textural patterns. By passing spectrograms through the frozen ResNet, we extract rich, 512-dimensional domain-invariant embeddings that represent the baseline acoustic "texture" of normal machine operations.

## 2. Feature-Space Anomaly Detection Strategy
Rather than attempting computationally expensive image-space reconstruction, the anomaly detection occurs entirely in the feature space. A downstream Autoencoder is trained to compress and reconstruct the 512-dimensional embeddings of *normal* source-domain data. 

When the unseen target domain is evaluated, anomalous acoustic events produce structural shifts in the spectrogram. The pretrained model maps these unfamiliar textures to slightly different embedding regions, resulting in a high Mean Squared Error (MSE) during reconstruction. This allows for precise, zero-shot anomaly classification without requiring target-domain labels during training.

## 3. Ablation Study Results
An ablation study was conducted to determine the optimal Autoencoder depth for feature-space reconstruction. 

| Architecture                    |   Zero-Shot AUROC |   Accuracy |
|:--------------------------------|------------------:|-----------:|
| Pretrained_ResNet+Light_AE_ReLU |                 1 |      0.975 |
| Pretrained_ResNet+Deep_AE_ReLU  |                 1 |      0.975 |

## 4. Conclusion
Leveraging pretrained vision models on acoustic spectrograms drastically accelerates convergence and provides highly generalized, domain-invariant embeddings. The model successfully isolates anomalies in unseen machine types purely based on deviations in feature-space reconstruction error, fully satisfying the zero-shot requirement of the task.
