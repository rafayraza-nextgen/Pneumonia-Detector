import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
IMG_SIZE   = 64
CHANNELS   = 3
BATCH_SIZE = 32
EPOCHS     = 30
LR_ADAM    = 0.001
LR_SGD     = 0.01
SEED       = 42
NUM_WEEKS  = 4

np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# ─────────────────────────────────────────────
# PHASE 1 — SYNTHETIC DATASET
# ─────────────────────────────────────────────
def generate_synthetic_dataset(n_normal=400, n_pneumonia=400):
    print("\n[Phase 1] Generating synthetic dataset...")

    X, y = [], []

    # Normal images
    for _ in range(n_normal):
        img = np.random.normal(loc=80, scale=20,
                               size=(IMG_SIZE, IMG_SIZE, CHANNELS))
        img = np.clip(img, 0, 255)
        X.append(img)
        y.append(0)

    # Pneumonia images
    for _ in range(n_pneumonia):
        img = np.random.normal(loc=80, scale=20,
                               size=(IMG_SIZE, IMG_SIZE, CHANNELS))

        for _ in range(np.random.randint(2, 6)):
            cx = np.random.randint(10, IMG_SIZE - 10)
            cy = np.random.randint(10, IMG_SIZE - 10)
            r = np.random.randint(5, 15)

            for i in range(max(0, cx - r), min(IMG_SIZE, cx + r)):
                for j in range(max(0, cy - r), min(IMG_SIZE, cy + r)):
                    if (i - cx)**2 + (j - cy)**2 < r**2:
                        img[i, j] += np.random.uniform(80, 140)

        img = np.clip(img, 0, 255)
        X.append(img)
        y.append(1)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    idx = np.random.permutation(len(X))
    return X[idx], y[idx]

# ─────────────────────────────────────────────
# PHASE 2 — SPLIT
# ─────────────────────────────────────────────
def preprocess_and_split(X, y):
    print("\n[Phase 2] Preprocessing...")

    X = X / 255.0

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    return X_train, X_val, X_test, y_train, y_val, y_test

# ─────────────────────────────────────────────
# CNN MODEL
# ─────────────────────────────────────────────
def build_cnn(optimizer_name="adam"):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation="relu", padding="same",
                      input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(64, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Conv2D(128, 3, activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    opt = optimizers.Adam(LR_ADAM) if optimizer_name == "adam" else optimizers.SGD(
        learning_rate=LR_SGD, momentum=0.9)

    model.compile(optimizer=opt,
                  loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model

# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train_model(model, X_train, y_train, X_val, y_val, label="Adam"):
    print(f"\n[Training with {label}]")

    cb = [
        callbacks.EarlyStopping(patience=5, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(patience=3, factor=0.5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=cb,
        verbose=1
    )

    return history

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":

    # ✅ ONLY DUMMY DATASET
    X, y = generate_synthetic_dataset()

    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_and_split(X, y)

    # Train Adam model
    model_adam = build_cnn("adam")
    hist_adam = train_model(model_adam, X_train, y_train, X_val, y_val, "Adam")

    # Train SGD model
    model_sgd = build_cnn("sgd")
    hist_sgd = train_model(model_sgd, X_train, y_train, X_val, y_val, "SGD")

    print("\nTraining complete.")