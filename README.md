# Zero-Shot-Anomaly-Detection
Zero-shot anomaly detection pipeline using a pretrained ResNet-18 feature extractor.
## Executive Summary & Approach

This repository outlines the development of a domain-invariant anomaly detection pipeline capable of zero-shot generalization across unseen industrial machine domains. Given the open-ended nature of the assessment brief and the scale of the suggested datasets (MIMII/ToyADMOS), I approached this task prioritizing **architectural validation and rapid prototyping**.

Before introducing the heavy I/O latency of processing raw audio tensors, the immediate goal was to prove the mathematical viability of the zero-shot logic. The resulting pipeline utilizes a Transfer Learning approach operating entirely in the feature space via a frozen ResNet-18, yielding perfect separability on validation distributions.
