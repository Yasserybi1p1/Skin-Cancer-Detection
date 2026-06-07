# 🔬 Skin Cancer Detection using HAM10000 Dataset

> **MobileNetV2 Transfer Learning for 7-Class Skin Lesion Classification**

A deep learning pipeline that classifies dermoscopic skin lesion images into 7 diagnostic categories using transfer learning with MobileNetV2. Built with **Keras 3** on a **PyTorch (CUDA) backend** for GPU-accelerated training on Windows.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Strategy](#training-strategy)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Output Files](#output-files)
- [Results](#results)
- [GPU Configuration](#gpu-configuration)
- [Troubleshooting](#troubleshooting)

---

## Overview

This project performs automated skin cancer classification on the **HAM10000** ("Human Against Machine with 10000 training images") dataset. It uses a two-phase transfer learning approach with MobileNetV2 pretrained on ImageNet, fine-tuned for dermatological image classification.

### Key Features

- **7-class skin lesion classification** from dermoscopic images
- **Two-phase transfer learning** (feature extraction → fine-tuning)
- **Lesion-level data splitting** to prevent data leakage
- **Class-weighted training** to handle severe class imbalance
- **GPU-accelerated** via Keras 3 + PyTorch CUDA backend
- **Comprehensive evaluation** with confusion matrix, ROC-AUC curves, and per-class metrics

---

## Dataset

**HAM10000** — a large collection of multi-source dermatoscopic images of common pigmented skin lesions.

- **Total Images**: ~10,015 dermoscopic images
- **Resolution**: Variable (resized to 224×224 for training)
- **Source**: [Harvard Dataverse / ISIC Archive](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T)

### Classes (7 Diagnostic Categories)

| Code | Full Name | Description |
|------|-----------|-------------|
| `akiec` | Actinic Keratoses | Pre-cancerous scaly patches |
| `bcc` | Basal Cell Carcinoma | Most common skin cancer |
| `bkl` | Benign Keratosis | Non-cancerous skin growths |
| `df` | Dermatofibroma | Benign fibrous skin nodules |
| `mel` | Melanoma | Most dangerous skin cancer |
| `nv` | Melanocytic Nevi | Common moles (benign) |
| `vasc` | Vascular Lesions | Blood vessel-related lesions |

> ⚠️ **Class Imbalance**: The dataset is heavily imbalanced — `nv` (moles) dominates with ~67% of all images, while `df` and `vasc` each have < 2%. Class weights are used during training to mitigate this.

### Data Splitting

Splitting is done at the **lesion level** (not image level) to prevent data leakage, since some lesions have multiple images (2–6 views):

| Split | Ratio | Purpose |
|-------|-------|---------|
| Train | 70% | Model training |
| Validation | 15% | Hyperparameter tuning & early stopping |
| Test | 15% | Final unbiased evaluation |

---

## Model Architecture

### Base Model: MobileNetV2

- **Pretrained on**: ImageNet (1000-class image classification)
- **Architecture**: Inverted residual blocks with linear bottlenecks
- **Total layers**: 155
- **Why MobileNetV2?**: Lightweight, efficient, strong transfer learning performance — well-suited for medical imaging on consumer GPUs

### Custom Classification Head

The pretrained MobileNetV2 base (without its original top layer) is extended with:

```
MobileNetV2 (frozen/partially frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dropout (0.3)
    ↓
Dense (128, ReLU)
    ↓
BatchNormalization
    ↓
Dropout (0.3)
    ↓
Dense (7, Softmax)  →  Output probabilities for 7 classes
```

### Parameter Counts

| Component | Parameters |
|-----------|-----------|
| MobileNetV2 base | ~2.2M (frozen in Phase 1) |
| Classification head | ~163K (always trainable) |
| **Total** | **~2.4M** |

---

## Training Strategy

### Two-Phase Transfer Learning

#### Phase 1 — Feature Extraction (Base Frozen)

| Setting | Value |
|---------|-------|
| Frozen layers | All MobileNetV2 layers |
| Trainable params | ~163K (head only) |
| Optimizer | Adam |
| Learning rate | `1e-3` |
| Epochs | Up to 10 |
| Early stopping | Patience = 5 (monitors `val_loss`) |
| LR reduction | Factor 0.5, patience 3, min `1e-7` |

**Purpose**: Train only the classification head so it learns to map MobileNetV2's pretrained ImageNet features to skin lesion classes — without disturbing the base weights.

#### Phase 2 — Fine-Tuning (Top Layers Unfrozen)

| Setting | Value |
|---------|-------|
| Unfrozen layers | Top 30 layers of MobileNetV2 |
| Trainable params | ~1M+ |
| Optimizer | Adam |
| Learning rate | `1e-5` (100× smaller than Phase 1) |
| Epochs | Up to 15 |
| Early stopping | Patience = 5 (monitors `val_loss`) |
| LR reduction | Factor 0.5, patience 3, min `1e-7` |

**Purpose**: Fine-tune the upper convolutional layers of MobileNetV2 to adapt the feature extraction specifically for dermoscopic images — using a very low learning rate to avoid catastrophic forgetting.

### Data Augmentation (Training Only)

Applied on-the-fly during training via `tf.data` pipeline:

| Augmentation | Details |
|-------------|---------|
| Horizontal flip | Random |
| Vertical flip | Random |
| Brightness | ±20% |
| Contrast | 80%–120% |
| Saturation | 80%–120% |
| Rotation | 0°, 90°, 180°, or 270° (random) |

### Class Imbalance Handling

Inverse-frequency **class weights** are computed using `sklearn.utils.class_weight.compute_class_weight('balanced')` and passed to `model.fit()`. This penalizes misclassifications of rare classes (e.g., `df`, `vasc`) more heavily than dominant classes (e.g., `nv`).

### Callbacks

| Callback | Purpose |
|----------|---------|
| `EarlyStopping` | Stops training when `val_loss` stops improving (patience=5), restores best weights |
| `ReduceLROnPlateau` | Halves learning rate when `val_loss` plateaus (patience=3) |
| `ModelCheckpoint` | Saves the best model (by `val_loss`) to disk |

---

## Installation

### Prerequisites

- **Python**: 3.13+
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3050 6GB)
- **CUDA Toolkit**: 12.4 (bundled with PyTorch)
- **OS**: Windows 10/11

### Step 1: Clone / Download the Project

```bash
cd "g:\Work\Cancer detection"
```

### Step 2: Create a Virtual Environment

```bash
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
# PyTorch with CUDA 12.4 support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Keras 3, TensorFlow (used only for tf.data pipeline), and ML libraries
pip install keras tensorflow numpy pandas matplotlib seaborn scikit-learn Pillow

# Jupyter support (if running the notebook)
pip install ipykernel
python -m ipykernel install --user --name=.venv --display-name=".venv"
```

### Step 4: Configure Keras Backend

Set PyTorch as the default Keras backend globally:

```bash
python -c "import os, json; p=os.path.expanduser('~/.keras/keras.json'); d=json.load(open(p)) if os.path.exists(p) else {}; d.update({'backend':'torch','floatx':'float32','epsilon':1e-07,'image_data_format':'channels_last'}); os.makedirs(os.path.dirname(p),exist_ok=True); json.dump(d,open(p,'w'),indent=4)"
```

Or manually edit `%USERPROFILE%\.keras\keras.json`:

```json
{
    "floatx": "float32",
    "epsilon": 1e-07,
    "backend": "torch",
    "image_data_format": "channels_last"
}
```

### Step 5: Download the Dataset

Download the HAM10000 dataset and place the files in the project directory:

```
Cancer detection/
├── HAM10000_images_part_1/     ← ~5,000 .jpg images
├── HAM10000_images_part_2/     ← ~5,000 .jpg images
└── HAM10000_metadata.csv       ← Diagnosis labels & metadata
```

### Verified Package Versions

| Package | Version |
|---------|---------|
| PyTorch | 2.6.0+cu124 |
| Keras | 3.14.1 |
| TensorFlow | 2.21.0 |
| scikit-learn | 1.9.0 |
| NumPy | 2.4.6 |
| Pandas | 3.0.3 |
| Pillow | 12.2.0 |
| Matplotlib | 3.10.9 |
| Seaborn | 0.13.2 |

---

## Usage

### Running the Notebook

1. Open `cancer_detection.ipynb` in VS Code or Jupyter Lab
2. Select the `.venv` kernel
3. **Restart the kernel** (important for backend initialization)
4. Run all cells sequentially

### GPU Verification

After running the GPU check cell, you should see:

```
Keras version: 3.14.1
Keras backend: torch
SUCCESS: GPU detected: NVIDIA GeForce RTX 3050 6GB Laptop GPU
Training will run on the GPU.
```

### Inference on a Single Image

```python
import os
os.environ["KERAS_BACKEND"] = "torch"
import keras
import numpy as np
from PIL import Image

# Load model
model = keras.models.load_model("output/skin_cancer_model.keras")

# Preprocess image
img = Image.open("path/to/skin_image.jpg").resize((224, 224))
img_array = np.array(img) / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
CLASS_NAMES = ['akiec', 'bcc', 'bkl', 'df', 'mel', 'nv', 'vasc']
prediction = model.predict(img_array, verbose=0)
predicted_class = CLASS_NAMES[np.argmax(prediction)]
confidence = np.max(prediction) * 100

print(f"Predicted: {predicted_class} ({confidence:.1f}%)")
```

---

## Project Structure

```
Cancer detection/
│
├── cancer_detection.ipynb      # Main notebook (training + evaluation)
├── .py                         # Notebook cell updater script
├── HAM10000_metadata.csv       # Dataset metadata (labels, patient info)
├── README.md                   # This documentation
│
├── HAM10000_images_part_1/     # Dermoscopic images (part 1)
├── HAM10000_images_part_2/     # Dermoscopic images (part 2)
│
├── output/                     # All generated outputs
│   ├── skin_cancer_model.keras         # Final trained model (~23 MB)
│   ├── best_model_phase1.keras         # Best Phase 1 checkpoint (~11 MB)
│   ├── best_model_phase2.keras         # Best Phase 2 checkpoint (~23 MB)
│   ├── label_mapping.json              # Class names & final metrics
│   ├── classification_report.txt       # Per-class precision/recall/F1
│   ├── class_distribution.png          # Class distribution bar chart
│   ├── sample_images.png               # Sample images per class
│   ├── training_curves.png             # Accuracy & loss curves
│   ├── confusion_matrix.png            # Confusion matrix heatmap
│   ├── roc_curves.png                  # ROC-AUC curves (One-vs-Rest)
│   └── sample_predictions.png          # Sample test predictions
│
└── .venv/                      # Python virtual environment
```

---

## Output Files

| File | Description |
|------|-------------|
| `skin_cancer_model.keras` | Final trained model (Phase 2), ready for inference |
| `best_model_phase1.keras` | Best checkpoint from feature extraction phase |
| `best_model_phase2.keras` | Best checkpoint from fine-tuning phase |
| `label_mapping.json` | Class names, label indices, and final metrics |
| `classification_report.txt` | Precision, recall, F1-score per class |
| `class_distribution.png` | Bar chart of class frequencies in the dataset |
| `sample_images.png` | 2 example images per class (14 total) |
| `training_curves.png` | Accuracy and loss over both training phases |
| `confusion_matrix.png` | 7×7 confusion matrix heatmap |
| `roc_curves.png` | Per-class ROC curves with AUC scores |
| `sample_predictions.png` | 10 sample test images with true vs predicted labels |

---

## Results

### Test Set Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | 63.5% |
| **Weighted ROC-AUC** | 0.872 |

### Per-Class Metrics

| Class | Precision | Recall | F1-Score | Support |
|-------|-----------|--------|----------|---------|
| akiec | 0.237 | 0.500 | 0.322 | 46 |
| bcc | 0.256 | 0.694 | 0.374 | 62 |
| bkl | 0.529 | 0.105 | 0.175 | 172 |
| df | 0.067 | 0.500 | 0.119 | 14 |
| mel | 0.394 | 0.444 | 0.417 | 151 |
| **nv** | **0.918** | **0.748** | **0.825** | **1053** |
| vasc | 0.205 | 0.857 | 0.330 | 21 |

> **Note**: The high ROC-AUC (0.872) indicates the model has strong ranking ability across classes, even though raw accuracy is affected by the extreme class imbalance. The dominant `nv` class achieves 0.825 F1. Rare classes like `df` (14 test samples) and `vasc` (21 test samples) show high recall but low precision due to limited training data.

---

## GPU Configuration

### Why PyTorch Backend?

**TensorFlow ≥ 2.11 dropped native GPU support on Windows.** Even with CUDA and cuDNN installed, TensorFlow will only run on CPU on native Windows. The options are:

1. ✅ **Use Keras 3 with PyTorch backend** (this project's approach)
2. Use WSL2 (Windows Subsystem for Linux)
3. Use TensorFlow-DirectML plugin

This project uses **Keras 3** which supports multiple backends. By setting the backend to **PyTorch**, the model runs on the GPU via PyTorch's native Windows CUDA support while keeping the same Keras API.

### Backend Configuration

The backend is set in two places for reliability:

1. **Global config** (`%USERPROFILE%\.keras\keras.json`): Sets `"backend": "torch"` as the system-wide default
2. **Notebook cell 1**: `os.environ["KERAS_BACKEND"] = "torch"` — must be set **before** importing Keras

### TensorFlow's Role

TensorFlow is still installed and used **only** for the `tf.data` data loading pipeline (`tf.data.Dataset`, `tf.image` augmentations). All model training, inference, and GPU compute runs through PyTorch.

---

## Troubleshooting

### "WARNING: No GPU detected" in the notebook

1. Verify PyTorch sees the GPU:
   ```python
   import torch
   print(torch.cuda.is_available())       # Should be True
   print(torch.cuda.get_device_name(0))   # Should show your GPU
   ```
2. Check that Keras backend is `torch` (not `tensorflow`):
   ```python
   import keras
   print(keras.config.backend())  # Should print "torch"
   ```
3. If it prints `tensorflow`, **restart the Jupyter kernel** and re-run from cell 1

### `AttributeError: 'tuple' object has no attribute 'as_list'`

This happens when using `tf.keras.backend.count_params()` with the PyTorch backend. Replace:
```python
# ❌ Broken on PyTorch backend
trainable_count = sum(tf.keras.backend.count_params(w) for w in model.trainable_weights)

# ✅ Works on all backends
trainable_count = sum(np.prod(w.shape) for w in model.trainable_weights)
```

### Keras backend not switching to `torch`

- Ensure `os.environ["KERAS_BACKEND"] = "torch"` is called **before** any `import keras` statement
- Check global config: `%USERPROFILE%\.keras\keras.json` should have `"backend": "torch"`
- Restart the kernel — the backend is locked at first import and cannot be changed mid-session

### Out of GPU Memory

If you encounter CUDA OOM errors on a 6GB GPU:
- Reduce `BATCH_SIZE` from 32 to 16 or 8
- Reduce `IMG_SIZE` from 224 to 128 (will affect accuracy)
- Close other GPU-consuming applications

---

## License

This project uses the HAM10000 dataset which is available under the [CC BY-NC-SA 4.0 License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

---

## Acknowledgments

- **HAM10000 Dataset**: Tschandl, P., Rosendahl, C. & Kittler, H. *The HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions.* Sci. Data 5, 180161 (2018).
- **MobileNetV2**: Sandler, M. et al. *MobileNetV2: Inverted Residuals and Linear Bottlenecks.* CVPR 2018.
- **Keras 3**: Multi-backend deep learning framework by François Chollet.
