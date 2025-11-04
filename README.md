# 🏃 Human Activity Recognition with Inertial Signals

> Supervised research project on automatic human activity classification from accelerometer and gyroscope data using the **MotionSense** dataset.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table of Contents

- [About](#-about)
- [Features](#-features)
- [Dataset](#-dataset)
- [Methodology](#-methodology)
- [Results](#-results)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Technologies](#-technologies)
- [Author](#-author)
- [License](#-license)

## 🎯 About

This project demonstrates the **superiority of deep learning approaches** for human activity classification from temporal inertial signals. We compare two complementary approaches:

1. **Random Forest** with manual statistical feature extraction
2. **1D CNN** with automatic temporal feature learning

The project also includes a **comprehensive optimization study**: overlap impact, feature selection, and hyperparameter tuning.

### 🎓 Academic Context

Project conducted as part of a supervised research activity, using the **MotionSense** dataset available on Kaggle. The objective is to classify 6 distinct human activities from smartphone inertial sensor data.

## ✨ Features

- ✅ **6 activity classification**: walking, jogging, upstairs/downstairs, sitting, standing
- ✅ **User-based split** to prevent data leakage
- ✅ **LOSO validation** (Leave-One-Subject-Out) for robust evaluation
- ✅ **240+ advanced features**: temporal, frequency (FFT), entropy, autocorrelation
- ✅ **Complete optimization**: overlap, feature selection, hyperparameters
- ✅ **CNN architecture with regularization** to prevent overfitting
- ✅ **Detailed visualizations**: confusion matrices, learning curves, distributions

## 📊 Dataset

**MotionSense Dataset**
- 📱 Source: Smartphone accelerometer and gyroscope data
- 👥 24 users
- 🏃 6 activities: `dws` (downstairs), `ups` (upstairs), `wlk` (walking), `jog` (jogging), `sit` (sitting), `std` (standing)
- 📏 Sensors: 12 dimensions (acceleration, rotation, gravity on x, y, z axes)
- 🔗 [Download on Kaggle](https://www.kaggle.com/malekzadeh/motionsense-dataset)

## 🔬 Methodology

### Data Preprocessing

```
1. Load CSV files per user/activity
2. Label grouping (e.g., sit_5, sit_13 → sit)
3. Sliding window segmentation:
   - Window size: 500 samples
   - Overlap: 80% (optimal found by GridSearch)
   - ~12,600 windows generated
```

### Approach 1: Random Forest

**Feature Extraction (240+ features)**
- Temporal statistics: mean, std, min, max, median, quartiles, variance, skewness, kurtosis
- Variations: total, mean, maximum variation
- Frequency domain: FFT (magnitude, dominant frequency)
- Shannon entropy
- Autocorrelation (lag-1, lag-5)
- Zero-crossings, energy, RMS

**Optimization**
- SelectKBest: k=250 optimal features
- GridSearchCV: 360 hyperparameter combinations tested
- 3-fold cross-validation

### Approach 2: 1D CNN

**Architecture**
```
Input (500, 12) 
    ↓
Conv1D(64) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv1D(128) → BatchNorm → MaxPool → Dropout(0.3)
    ↓
Conv1D(256) → BatchNorm → MaxPool → Dropout(0.4)
    ↓
Conv1D(256) → BatchNorm → GlobalAvgPool → Dropout(0.5)
    ↓
Dense(128) → Dropout(0.5) → Dense(6, softmax)
```

**Regularization**
- L2 regularization (0.001)
- Batch Normalization
- Progressive Dropout (0.3 → 0.5)
- Early Stopping (patience=15)
- ReduceLROnPlateau

## 🏆 Results

### Comparative Performance

| Model | Accuracy | Configuration |
|-------|----------|---------------|
| **Random Forest (70/30 Split)** | **94.65%** | 132 baseline features |
| **Random Forest (LOSO)** | **97.16%** | 132 baseline features |
| **Optimized Random Forest** | **~98%** | 250 features + GridSearch |
| **1D CNN** | **98.25%** | End-to-end learning |

### Analysis

✅ **Consistency**: 70/30 Split < LOSO < CNN (logical progression)  
✅ **Generalization**: LOSO validation confirms robust cross-user performance  
✅ **CNN Improvement**: +3.6 points vs RF baseline through automatic temporal features  
✅ **No Overfitting**: Train/validation gap <2% with regularization  

### Confusion Matrix (CNN)

Most frequent confusions are logical:
- `wlk` ↔ `jog` (similar activities)
- `sit` ↔ `std` (transitions)
- `ups` ↔ `dws` (vertical movements)

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
pip
```

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/your-username/human-activity-recognition.git
cd human-activity-recognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### requirements.txt

```txt
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
tensorflow>=2.8.0
scipy>=1.7.0
```

## 💻 Usage

### 1. Download the Dataset

Download the [MotionSense Dataset](https://www.kaggle.com/malekzadeh/motionsense-dataset) and place it in the `data/` folder.

Expected structure:
```
data/
├── dws/
│   ├── sub_1.csv
│   ├── sub_2.csv
│   └── ...
├── ups/
├── wlk/
├── jog/
├── sit/
└── std/
```

### 2. Run the Notebook

```bash
jupyter notebook motionsense_classification.ipynb
```

### 3. Or Use Python Scripts

```bash
# Train Random Forest
python train_random_forest.py --data_path data/ --window_size 500 --overlap 0.8

# Train CNN
python train_cnn.py --data_path data/ --window_size 500 --epochs 100

# Complete optimization
python optimize.py --data_path data/
```

## 📁 Project Structure

```
human-activity-recognition/
│
├── data/                          # Dataset (not included)
│   ├── dws/
│   ├── ups/
│   └── ...
│
├── notebooks/
│   └── motionsense_classification.ipynb   # Main notebook
│
├── src/
│   ├── data_loading.py           # Data loading
│   ├── preprocessing.py          # Window segmentation
│   ├── feature_extraction.py    # Feature extraction
│   ├── models.py                 # RF and CNN architectures
│   └── optimization.py           # GridSearch and selection
│
├── models/                       # Saved models
│   ├── rf_model.pkl
│   ├── cnn_model.h5
│   └── scaler.pkl
│
├── results/                      # Visualizations and reports
│   ├── confusion_matrices/
│   ├── training_curves/
│   └── optimization_results/
│
├── requirements.txt              # Dependencies
├── README.md                     # This file
└── LICENSE                       # MIT License
```

## 🛠️ Technologies

- **Python 3.8+**: Main language
- **NumPy & Pandas**: Data manipulation
- **Scikit-learn**: Random Forest, preprocessing, metrics
- **TensorFlow/Keras**: 1D CNN
- **Matplotlib & Seaborn**: Visualizations
- **SciPy**: FFT, advanced statistics

## 📈 Future Improvements

- [ ] Data augmentation (temporal rotation, noise addition)
- [ ] ResNet 1D architecture
- [ ] Attention mechanisms / Transformers
- [ ] Ensemble methods (RF + CNN)
- [ ] Deployment with Flask/FastAPI
- [ ] Real-time mobile application

## 📚 References

1. Malekzadeh, M., et al. (2019). "Mobile Sensor Data Anonymization"
2. Goodfellow, I., et al. (2016). "Deep Learning" - MIT Press
3. Breiman, L. (2001). "Random Forests" - Machine Learning

## 👨‍💻 Author

Michael(Me)
- LinkedIn: [Your Profile]([https://linkedin.com/in/your-profil](https://www.linkedin.com/in/ivan-komi-25397028a)

## 🙏 Acknowledgments

- MotionSense Dataset by Mohammad Malekzadeh
- Supervising Professor: Julien Maitre,PhD
- Kaggle Community

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

⭐ **If you found this project helpful, please give it a star!** ⭐

## 📞 Contact

For questions or suggestions, feel free to open an [issue](https://github.com/your-username/human-activity-recognition/issues) or contact me directly.

---

*Supervised research project - 2024*
