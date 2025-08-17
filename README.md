# Recyclable_Object_Classification_using_CNN
## 📌 Overview
This project is a **waste classification system** that uses a **Convolutional Neural Network (CNN)** to classify waste as **Organic (O)** or **Recyclable (R)**.  
It aims to assist in proper waste segregation for improved recycling efficiency and environmental sustainability.

---

## 🎯 Objective
- Automate the process of waste identification.
- Achieve high accuracy in classifying waste into two categories: **Organic** and **Recyclable**.
- Provide a model that can be integrated into real-time applications like mobile or web apps.

---

## 🛠️ Technology Stack
- **Python** (3.11)
- **TensorFlow / Keras**
- **NumPy**
- **Pillow**
- **SciPy**
- **Streamlit** (for UI)

---

## 📂 Dataset
Initially, a dataset was used which led to poor model performance (e.g., plastics being classified as organic).  
To improve accuracy, we trained the model on a **better dataset**:
- Source: [Waste Classification Dataset - Kaggle](https://www.kaggle.com/datasets/techsash/waste-classification-data)
- Structure:
 DATASET/
 TRAIN/
 O/ # Organic
 R/ # Recyclable
 TEST/
 O/
 R/
- **Total Images:** ~25,000

---

## ⚙️ Model Architecture
- **Convolutional layers** for feature extraction.
- **MaxPooling layers** for spatial dimension reduction.
- **Flatten layer** to convert features into a vector.
- **Dense layers** for classification.
- **Softmax activation** for output probabilities.

---

## 🚧 Problems Faced & Solutions

| Problem | Cause | Solution |
|---------|-------|----------|
| TensorFlow OverflowError (`Python int too large to convert to C long`) | Compatibility issue with certain TF backends | Installed Python 3.11 and disabled oneDNN optimizations using `TF_ENABLE_ONEDNN_OPTS=0` |
| Missing dependencies (`scipy` error) | New virtual environment did not have all required libraries | Installed missing packages using `pip install scipy` |
| GPU not being used | TensorFlow-CPU was installed | Installed `tensorflow` with GPU support (optional) |
| Poor accuracy with first dataset | Dataset had noise/mislabeled images | Switched to Kaggle Waste Classification dataset and retrained |
| Virtual environment creation errors in `Program Files` | Restricted system folder permissions | Created `venv` in the project directory instead |

---

## 🚀 How to Run

### 1. Create a virtual environment
python -m venv venv

### 2. Activate it
venv\Scripts\activate

### 3.Install dependencies
pip install -r requirements.txt
pip install scipy

### 4. Train the model
python train_model.py

### 5. Run the Streamlitapp
streamlit run main.py


## Result
Model saved as: best_model.keras
