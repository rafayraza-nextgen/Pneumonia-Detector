
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
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# =============================================
# GLOBAL CONFIG
# =============================================
IMG_SIZE = 150          # Better quality than 64 for real X-rays
CHANNELS = 3
BATCH_SIZE = 32
EPOCHS = 50             # Increased with EarlyStopping
LR_ADAM = 0.001
LR_SGD = 0.01
SEED = 42
NUM_WEEKS = 4

np.random.seed(SEED)
tf.random.set_seed(SEED)

OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

# =============================================
# PHASE 1 — LOAD REAL DATASET
# =============================================
def load_real_dataset(data_dir="chest_xray"):
    print("\n[Phase 1] Loading real Chest X-Ray dataset...")
    images = []
    labels = []
    
    for split in ['train', 'test', 'val']:
        for class_name, label in [('NORMAL', 0), ('PNEUMONIA', 1)]:
            class_dir = os.path.join(data_dir, split, class_name)
            if not os.path.exists(class_dir):
                print(f"    Warning: {class_dir} not found!")
                continue
                
            files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.jpeg', '.jpg', '.png'))]
            
            for fname in files:
                path = os.path.join(class_dir, fname)
                try:
                    img = load_img(path, target_size=(IMG_SIZE, IMG_SIZE), color_mode='rgb')
                    img_array = img_to_array(img)
                    images.append(img_array)
                    labels.append(label)
                except Exception as e:
                    print(f"    Error loading {fname}: {e}")
    
    X = np.array(images, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    
    # Shuffle
    idx = np.random.permutation(len(X))
    X, y = X[idx], y[idx]
    
    n_normal = np.sum(y == 0)
    n_pneumonia = np.sum(y == 1)
    print(f"    Loaded {len(X)} images | Normal: {n_normal} | Pneumonia: {n_pneumonia}")
    return X, y


# =============================================
# PHASE 2 — PREPROCESSING & SPLITTING
# =============================================
def preprocess_and_split(X, y):
    print("\n[Phase 2] Preprocessing and splitting dataset...")
    X = X / 255.0  # Normalize to [0, 1]

    # 70% Train, 15% Val, 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, random_state=SEED, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=SEED, stratify=y_temp)

    print(f"    Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")

    # Temporal weekly batches from test set
    weekly_batches = []
    week_size = len(X_test) // NUM_WEEKS
    for w in range(NUM_WEEKS):
        start = w * week_size
        end = start + week_size if w < NUM_WEEKS - 1 else len(X_test)
        weekly_batches.append((X_test[start:end], y_test[start:end]))

    print(f"    Created {NUM_WEEKS} weekly temporal batches")
    return X_train, X_val, X_test, y_train, y_val, y_test, weekly_batches


# =============================================
# PHASE 3 — CNN ARCHITECTURE
# =============================================
def build_cnn(optimizer_name="adam"):
    print(f"\n[Phase 3] Building CNN with {optimizer_name.upper()} optimizer...")
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", padding="same", input_shape=(IMG_SIZE, IMG_SIZE, CHANNELS)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(1, activation="sigmoid")
    ])

    opt = (optimizers.Adam(learning_rate=LR_ADAM) if optimizer_name == "adam" 
           else optimizers.SGD(learning_rate=LR_SGD, momentum=0.9))

    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    return model


# =============================================
# PHASE 4 — TRAINING WITH AUGMENTATION
# =============================================
def train_model(model, X_train, y_train, X_val, y_val, label="Adam"):
    print(f"\n[Phase 4] Training with {label} optimizer...")

    # Data Augmentation
    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_datagen = ImageDataGenerator()

    train_generator = train_datagen.flow(X_train, y_train, batch_size=BATCH_SIZE, seed=SEED)
    val_generator   = val_datagen.flow(X_val,   y_val,   batch_size=BATCH_SIZE, seed=SEED)

    callbacks_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1)
    ]

    steps_per_epoch = len(X_train) // BATCH_SIZE
    validation_steps = len(X_val) // BATCH_SIZE

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_generator,
        validation_steps=validation_steps,
        epochs=EPOCHS,
        callbacks=callbacks_list,
        verbose=1
    )
    
    best_val_acc = max(history.history['val_accuracy'])
    print(f"    Training complete. Best val_accuracy: {best_val_acc:.4f}")
    return history


# =============================================
# PHASE 5 — EVALUATION & PLOTS (Your original functions - slightly cleaned)
# =============================================

def plot_training_curves(hist_adam, hist_sgd):
    print("\n[Phase 5a] Plotting training curves...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("CNN Training Performance — Chest X-Ray Pneumonia Detection", fontsize=14, fontweight="bold")
    
    styles = [(hist_adam, "Adam", "#2563EB", "#93C5FD"), (hist_sgd, "SGD", "#16A34A", "#86EFAC")]
    
    for col, (hist, name, c_train, c_val) in enumerate(styles):
        ep = range(1, len(hist.history["accuracy"]) + 1)
        axes[0][col].plot(ep, hist.history["accuracy"], color=c_train, lw=2, label="Train")
        axes[0][col].plot(ep, hist.history["val_accuracy"], color=c_val, lw=2, linestyle="--", label="Validation")
        axes[0][col].set_title(f"{name} — Accuracy")
        axes[0][col].set_xlabel("Epoch"); axes[0][col].set_ylabel("Accuracy")
        axes[0][col].legend(); axes[0][col].grid(alpha=0.3); axes[0][col].set_ylim(0, 1.05)

        axes[1][col].plot(ep, hist.history["loss"], color=c_train, lw=2, label="Train loss")
        axes[1][col].plot(ep, hist.history["val_loss"], color=c_val, lw=2, linestyle="--", label="Val loss")
        axes[1][col].set_title(f"{name} — Loss")
        axes[1][col].set_xlabel("Epoch"); axes[1][col].set_ylabel("Loss")
        axes[1][col].legend(); axes[1][col].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "01_training_curves.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("    Saved → 01_training_curves.png")


def plot_confusion_matrix(model, X_test, y_test, optimizer_name):
    print(f"\n[Phase 5b] Confusion Matrix ({optimizer_name})...")
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
    cm = confusion_matrix(y_test, y_pred)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Pneumonia"])
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"Confusion Matrix — {optimizer_name}")
    
    report = classification_report(y_test, y_pred, target_names=["Normal", "Pneumonia"], digits=4)
    print(f"\nClassification Report ({optimizer_name}):\n{report}")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"02_confusion_matrix_{optimizer_name.lower()}.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print(f"    Saved → 02_confusion_matrix_{optimizer_name.lower()}.png")


def plot_weekly_performance(model_adam, model_sgd, weekly_batches):
    print("\n[Phase 5c] Weekly Temporal Performance...")
    weeks = [f"Week {i+1}" for i in range(NUM_WEEKS)]
    acc_adam, acc_sgd = [], []
    
    for Xw, yw in weekly_batches:
        _, a_adam = model_adam.evaluate(Xw, yw, verbose=0)
        _, a_sgd = model_sgd.evaluate(Xw, yw, verbose=0)
        acc_adam.append(a_adam)
        acc_sgd.append(a_sgd)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(NUM_WEEKS)
    width = 0.35
    ax.bar(x - width/2, acc_adam, width, label="Adam", color="#2563EB", alpha=0.85)
    ax.bar(x + width/2, acc_sgd,  width, label="SGD",  color="#16A34A", alpha=0.85)
    
    ax.set_title("Model Accuracy Across Weekly Temporal Batches", fontweight="bold")
    ax.set_xlabel("Temporal Batch (Simulated Week)")
    ax.set_ylabel("Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(weeks)
    ax.set_ylim(0, 1.1)
    ax.axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Random (0.5)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "03_weekly_performance.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("    Saved → 03_weekly_performance.png")


def plot_sample_predictions(model, X_test, y_test, n=10):
    print("\n[Phase 5d] Sample Predictions...")
    y_pred_prob = model.predict(X_test[:n], verbose=0).flatten()
    y_pred = (y_pred_prob > 0.5).astype(int)
    labels = ["Normal", "Pneumonia"]
    
    fig, axes = plt.subplots(2, 5, figsize=(15, 7))
    fig.suptitle("Sample Predictions (Adam Model)", fontsize=14, fontweight="bold")
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(X_test[i], cmap="gray")
        correct = y_pred[i] == y_test[i]
        color = "green" if correct else "red"
        ax.set_title(f"True: {labels[y_test[i]]}\nPred: {labels[y_pred[i]]} ({y_pred_prob[i]:.3f})",
                     color=color, fontsize=9)
        ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "04_sample_predictions.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("    Saved → 04_sample_predictions.png")


def plot_interpretation(hist_adam):
    print("\n[Phase 5e] Interpretation Summary Chart...")
    # (Your original function - kept as is with minor cleanup)
    fig = plt.figure(figsize=(14, 6))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.4)
    
    ax1 = fig.add_subplot(gs[0, 0])
    ep = range(1, len(hist_adam.history["accuracy"]) + 1)
    ax1.plot(ep, hist_adam.history["accuracy"], "#2563EB", lw=2, label="Train")
    ax1.plot(ep, hist_adam.history["val_accuracy"], "#93C5FD", lw=2, linestyle="--", label="Validation")
    
    val_acc = hist_adam.history["val_accuracy"]
    best_ep = int(np.argmax(val_acc)) + 1
    ax1.axvline(best_ep, color="orange", linestyle=":", label=f"Best epoch {best_ep}")
    
    ax1.set_title("Accuracy Interpretation")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Accuracy")
    ax1.legend(); ax1.grid(alpha=0.3); ax1.set_ylim(0, 1.05)
    
    ax2 = fig.add_subplot(gs[0, 1])
    metrics = ["Max Train Acc", "Max Val Acc", "Min Train Loss", "Min Val Loss"]
    values = [
        max(hist_adam.history["accuracy"]),
        max(hist_adam.history["val_accuracy"]),
        min(hist_adam.history["loss"]),
        min(hist_adam.history["val_loss"])
    ]
    colors = ["#2563EB", "#93C5FD", "#16A34A", "#86EFAC"]
    bars = ax2.bar(metrics, values, color=colors, alpha=0.9)
    for bar, v in zip(bars, values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f"{v:.3f}", ha="center")
    
    ax2.set_title("Key Metrics (Adam)")
    ax2.set_ylabel("Score")
    ax2.grid(axis="y", alpha=0.3)
    
    plt.suptitle("Result Interpretation Summary", fontsize=14, fontweight="bold")
    plt.savefig(os.path.join(OUT_DIR, "05_interpretation_summary.png"), dpi=200, bbox_inches="tight")
    plt.close()
    print("    Saved → 05_interpretation_summary.png")


def plot_architecture():
    print("\n[Phase 5f] Plotting CNN Architecture Diagram...")
    # (Your original architecture plot kept - it's very good)
    fig, ax = plt.subplots(figsize=(15, 5))
    ax.set_xlim(0, 16); ax.set_ylim(0, 4); ax.axis("off")
    ax.set_facecolor("#F8FAFC")
    fig.patch.set_facecolor("#F8FAFC")
    
    layers_info = [
        ("Input\n150×150×3", "#DBEAFE", 0.6),
        ("Conv2D\n32 filters", "#BFDBFE", 0.9),
        ("MaxPool\n2×2", "#93C5FD", 0.7),
        ("Conv2D\n64 filters", "#60A5FA", 1.1),
        ("MaxPool\n2×2", "#3B82F6", 0.7),
        ("Conv2D\n128 filters", "#2563EB", 1.2),
        ("MaxPool\n2×2", "#1D4ED8", 0.7),
        ("Flatten", "#7C3AED", 0.6),
        ("Dense 256\n+ Dropout 0.5", "#8B5CF6", 1.0),
        ("Sigmoid\nOutput", "#10B981", 0.65),
    ]
    
    box_w = 1.2; gap = 0.3; start_x = 0.3
    for i, (label, color, height) in enumerate(layers_info):
        x = start_x + i * (box_w + gap)
        y_box = (4 - height * 2) / 2
        rect = plt.Rectangle((x, y_box), box_w, height*2, facecolor=color, edgecolor="white", linewidth=2)
        ax.add_patch(rect)
        ax.text(x + box_w/2, 2, label, ha="center", va="center", fontsize=8, fontweight="bold", color="white", wrap=True)
        if i < len(layers_info)-1:
            ax.annotate("", xy=(x+box_w+gap, 2), xytext=(x+box_w, 2),
                        arrowprops=dict(arrowstyle="->", color="#475569", lw=1.5))
    
    ax.set_title("CNN Architecture — Chest X-Ray Pneumonia Detection", fontsize=14, fontweight="bold", pad=20)
    plt.savefig(os.path.join(OUT_DIR, "06_cnn_architecture.png"), dpi=200, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print("    Saved → 06_cnn_architecture.png")


# =============================================
# MAIN
# =============================================
def main():
    print("="*70)
    print("   CNN Chest X-Ray Pneumonia Detection")
    print("   Dawood University of Engineering and Technology")
    print("="*70)

    X, y = load_real_dataset("chest_xray")
    
    X_train, X_val, X_test, y_train, y_val, y_test, weekly = preprocess_and_split(X, y)

    # Train with Adam
    model_adam = build_cnn("adam")
    model_adam.summary()
    hist_adam = train_model(model_adam, X_train, y_train, X_val, y_val, label="Adam")

    # Train with SGD
    model_sgd = build_cnn("sgd")
    hist_sgd = train_model(model_sgd, X_train, y_train, X_val, y_val, label="SGD")

    # Generate all plots
    plot_architecture()
    plot_training_curves(hist_adam, hist_sgd)
    plot_confusion_matrix(model_adam, X_test, y_test, "Adam")
    plot_confusion_matrix(model_sgd,  X_test, y_test, "SGD")
    plot_weekly_performance(model_adam, model_sgd, weekly)
    plot_sample_predictions(model_adam, X_test, y_test)
    plot_interpretation(hist_adam)

    # Final Results
    print("\n" + "="*70)
    _, test_acc_adam = model_adam.evaluate(X_test, y_test, verbose=0)
    _, test_acc_sgd  = model_sgd.evaluate(X_test, y_test, verbose=0)
    print(f"Final Test Accuracy — Adam : {test_acc_adam:.4f}")
    print(f"Final Test Accuracy — SGD  : {test_acc_sgd:.4f}")
    print("="*70)
    print(f"\nAll plots saved to '{OUT_DIR}' folder")


if __name__ == "__main__":
    main()