# 🫁 CNN Chest X-Ray Pneumonia Detection

A Deep Learning project using Convolutional Neural Networks (CNNs) to classify chest X-ray images into:

- **Normal**
- **Pneumonia**

The project supports both:
- 🧪 Synthetic dataset (for fast testing and debugging)
- 🏥 Real Chest X-ray dataset (Kaggle)

It also compares **Adam vs SGD optimizers** and provides full evaluation visualizations.

---

## 🚀 Features

- CNN-based image classification
- Synthetic dataset generator for quick experimentation
- Real dataset loader (Kaggle Chest X-ray dataset)
- Data augmentation for better generalization
- Adam vs SGD optimizer comparison
- Confusion matrix + classification report
- Weekly (temporal) performance simulation
- Training curves visualization
- Sample prediction visualization
- CNN architecture diagram

---

## 🧠 Model Architecture

The CNN consists of:

- Conv2D (32 filters)
- Conv2D (64 filters)
- Conv2D (128 filters)
- Batch Normalization
- MaxPooling layers
- Fully Connected layer (256 neurons)
- Dropout (0.5)
- Output layer (Sigmoid activation)

---

## 📂 Project Structure
