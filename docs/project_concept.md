# Project Concept

## Project Title

GAN-Based Synthetic Data Generation for Public Procurement Corruption Risk Detection

## Dissertation Connection

This project is directly connected with the dissertation topic:

**Machine Learning Based Early Warning for Corruption Risk in Public Procurement**

The dissertation focuses on the development of an early warning approach for identifying corruption risks in public procurement. The GAN project supports this research by generating synthetic procurement records that can be used for risk analysis, model testing, and data augmentation.

## Research Problem

Public procurement data often contains sensitive, incomplete, imbalanced, or limited records. This makes it difficult to build and test machine learning models for corruption risk detection. Generative Adversarial Networks can help by learning patterns from real procurement data and producing synthetic records with similar statistical characteristics.

## Object of Research

The object of research is public procurement data and corruption risk detection processes.

## Subject of Research

The subject of research is the application of Generative Adversarial Networks for synthetic data generation in public procurement risk analysis.

## Project Goal

The goal of the project is to develop a GAN-based prototype that generates synthetic public procurement data and demonstrates how generated data can support corruption risk detection.

## Project Tasks

1. Analyze the structure of public procurement data.
2. Select features related to corruption risk detection.
3. Prepare and preprocess procurement data for machine learning.
4. Design a GAN model for tabular procurement data generation.
5. Train the GAN model using Google Colab to reduce local computer load.
6. Generate synthetic procurement records.
7. Compare real and synthetic data using statistical and visual methods.
8. Explain how synthetic data can support early warning systems for corruption risks.

## Hypothesis

If a GAN model is trained on public procurement data, then it can generate synthetic procurement records with similar statistical patterns, which can be used to support corruption risk analysis and improve machine learning experiments when real data is limited or imbalanced.

## Practical Value

The practical value of the project is that synthetic procurement data can be used for:

- testing corruption risk detection algorithms;
- increasing the amount of training data;
- reducing dependence on sensitive real procurement records;
- supporting early-stage ML experiments;
- demonstrating risk patterns in public procurement.

## Expected Result

The expected result is a working prototype that:

1. loads and preprocesses procurement data;
2. trains a GAN model in Google Colab;
3. generates synthetic procurement records;
4. compares real and generated data;
5. explains the relevance of GAN-generated data for corruption risk detection.

## Tools and Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- PyTorch
- Google Colab
- Matplotlib
- GitHub

## Main Dataset

The project will use public procurement data related to ProZorro or another open procurement dataset. The final dataset will be reduced and preprocessed to avoid high load on the local computer.

## Local Computer Load Reduction Strategy

To reduce the load on the MacBook:

1. raw datasets will be stored on the external SSD;
2. large files will not be pushed to GitHub;
3. heavy model training will be done in Google Colab;
4. locally, only lightweight scripts, configuration files, and documentation will be maintained;
5. generated models and large outputs will be ignored by Git.
