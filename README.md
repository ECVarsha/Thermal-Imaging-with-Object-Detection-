# Thermal Image-Based Object Classification using EfficientNetB0

A deep learning project to classify thermal images into categories like **Car**, **Cat**, and **Man** using transfer learning with EfficientNetB0.

---

## 📁 Dataset

**Source:** [Kaggle – Thermal Image Dataset for Object Classification](https://www.kaggle.com/datasets/iamsouravbanerjee/thermal-image-dataset-for-object-classification)

- 3 Classes: `Car`, `Cat`, `Man`
- Thermal grayscale images
- Format: `.jpg`


---

## 🛠️ Tech Stack

| Component        | Details                                  |
|------------------|-------------------------------------------|
| **Model**        | EfficientNetB0 (Transfer Learning)        |
| **Framework**    | TensorFlow + Keras                        |
| **Language**     | Python 3                                  |
| **Data Aug.**    | `ImageDataGenerator`                      |
| **Callbacks**    | EarlyStopping, ReduceLROnPlateau, Checkpoint |

---

## 🚀 How to Run

###  1. Clone the Repository

```bash
git clone https://github.com/your-username/thermal-object-classification.git
cd thermal-object-classification

### 2. Install Dependencies

```bash
pip install -r requirements.txt

### 3. Prepare the Dataset

```bash
data/Thermal Image Dataset/train/
data/Thermal Image Dataset/val/

### 4. Train the Model

```bash
python train_model.py

###
Model Architecture
Base: EfficientNetB0 (pretrained on ImageNet)
Top Layers:
  - GlobalAveragePooling2D
  - Dropout (0.3)
  - Dense layer with softmax (3 classes)
Callbacks:
  - EarlyStopping (patience=3)
  - ReduceLROnPlateau (patience=2)
  - ModelCheckpoint (best model saved)

