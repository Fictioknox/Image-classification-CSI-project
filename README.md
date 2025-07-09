# 🌿 Traditional Image Classification using Handcrafted Features

This project classifies plant leaf images into four categories — `healthy`, `rust`, `scab`, and `multiple diseases` — using **traditional machine learning** techniques and **handcrafted features**. No deep learning or transfer learning is used.

---

## 📌 Table of Contents

- [📌 Table of Contents](#-table-of-contents)
- [🚀 Project Overview](#-project-overview)
- [📁 Dataset](#-dataset)
- [🧪 Feature Extraction](#-feature-extraction)
- [🤖 Machine Learning Models](#-machine-learning-models)
- [📊 Evaluation Metrics](#-evaluation-metrics)
- [🛠️ How to Run](#️-how-to-run)
- [📈 Results](#-results)

---

## 🚀 Project Overview

- 🎯 **Goal**: Multi-class classification of plant leaf images using handcrafted image features.
- ⚙️ **Approach**:
  - No CNNs or transfer learning
  - Handcrafted features only (HOG, color histogram, texture)
  - Classical ML models (SVM, Random Forest)

---

## 📁 Dataset

- Images and labels sourced from the [Plant Pathology 2020 dataset](https://www.kaggle.com/c/plant-pathology-2020-fgvc7).
- Each image is labeled as one of:
  - `healthy`
  - `multiple_diseases`
  - `rust`
  - `scab`

---

## 🧪 Feature Extraction

Extracted handcrafted features include:

- 🔹 **HOG (Histogram of Oriented Gradients)** – captures shape and edges  
- 🔹 **Color Histogram** – captures color distribution across R, G, B channels  
- 🔹 **Texture Features** – based on Sobel edge magnitudes

```python
features = np.concatenate([hog_features, color_hist, texture_features])
