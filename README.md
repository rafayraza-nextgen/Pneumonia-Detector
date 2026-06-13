# CNN Chest X-Ray Pneumonia Detection

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


cnn-xray/
│
├── cnn_synthetic.py # Synthetic dataset version
├── cnn_real.py # Real dataset version (Kaggle)
├── outputs/ # Generated plots & results
└── README.md


---

## 📊 Dataset

### 🧪 Synthetic Dataset
- Automatically generated X-ray-like images
- No external download required
- Used for testing pipeline

### 🏥 Real Dataset (Kaggle)
Download from:
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Folder structure:

chest_xray/
├── train/
│ ├── NORMAL/
│ └── PNEUMONIA/
├── test/
│ ├── NORMAL/
│ └── PNEUMONIA/
└── val/


---

## ⚙️ Installation

```bash
git clone https://github.com/your-username/cnn-xray.git
cd cnn-xray
pip install -r requirements.txt
▶️ How to Run
🧪 Synthetic Version
python cnn_synthetic.py
🏥 Real Dataset Version

Make sure dataset path is correct:

load_real_dataset("chest_xray")

Then run:

python cnn_real.py
📈 Outputs

All outputs are saved in:

outputs/

Generated visualizations:

Training Accuracy & Loss curves
Confusion Matrix (Adam & SGD)
Weekly performance comparison
Sample predictions
CNN architecture diagram
Model interpretation summary
📊 Results Summary
Adam optimizer → Faster convergence, higher accuracy
SGD optimizer → Slower but stable learning
Data augmentation improves generalization significantly
CNN effectively detects pneumonia patterns in X-rays
📉 Evaluation Metrics
Accuracy
Precision
Recall
F1-score
Confusion Matrix
Temporal performance (weekly batches)
🔬 Key Insights
Data augmentation reduces overfitting
Adam performs better for this dataset
CNN learns opacity patterns linked to pneumonia
Synthetic dataset is useful for debugging pipeline
📌 Future Improvements
Add Grad-CAM explainability
Use pretrained models (ResNet, EfficientNet)
Deploy using Flask / FastAPI
Convert model to TensorFlow Lite
Improve class imbalance handling
👨‍💻 Author

Developed for academic Deep Learning research and portfolio project.

📜 License

This project is open-source and free to use for educational purposes.

⭐ Support

If you like this project, consider giving it a ⭐ on GitHub!


---

If you want next step, I can also:
- :contentReference[oaicite:0]{index=0}
- :contentReference[oaicite:1]{index=1}
- or :contentReference[oaicite:2]{index=2}

Just tell me 👍
