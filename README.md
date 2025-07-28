# Infant Cry Classification with CNN & EmbedAug

## 📌 Project Overview
This project classifies infant cries into categories like **asphyxia, deaf, hunger, normal, and pain** using a CNN-based architecture enhanced with **EmbedAug** and **audio waveform-level augmentation**. It leverages **Mel-spectrogram features**, **weighted loss**, and **early stopping** for improved performance. Achieved ~81% accuracy on the Baby Chillanto dataset.

---

## ✅ Features
- CNN-based classifier for audio classification  
- **EmbedAug** (feature-level augmentation)  
- **Waveform-level augmentations** (noise, time-shift, volume scaling)  
- **Class-weighted loss** to handle imbalance  
- **Early stopping & best model saving**  
- **Evaluation metrics**: Confusion Matrix, F1-score, CSV export  

---

## 🗂 Dataset
Dataset should be structured as:
data/
├── asphyxia/
├── deaf/
├── hunger/
├── normal/
└── pain/

---

Each folder contains `.wav` files for its respective class.

---

## 🛠 Installation
```bash
# Clone the repository
git clone https://github.com/KaRTiK0821/EmbedAug-Classification.git
cd infant-cry-classification

# Install dependencies
pip install -r requirements.txt
