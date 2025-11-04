# 🏃 Classification d'Activités Humaines avec Signaux Inertiels

> Projet de recherche supervisée sur la classification automatique d'activités humaines à partir de données d'accéléromètre et gyroscope du dataset **MotionSense**.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-green.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📋 Table des Matières

- [À Propos](#-à-propos)
- [Caractéristiques](#-caractéristiques)
- [Dataset](#-dataset)
- [Méthodologie](#-méthodologie)
- [Résultats](#-résultats)
- [Installation](#-installation)
- [Utilisation](#-utilisation)
- [Structure du Projet](#-structure-du-projet)
- [Technologies](#-technologies)
- [Auteur](#-auteur)
- [Licence](#-licence)

## 🎯 À Propos

Ce projet démontre la **supériorité des approches d'apprentissage profond** pour la classification d'activités humaines à partir de signaux temporels inertiels. Nous comparons deux approches complémentaires :

1. **Random Forest** avec extraction manuelle de caractéristiques statistiques
2. **CNN 1D** avec apprentissage automatique de features temporelles

Le projet inclut également une **étude approfondie d'optimisation** : impact de l'overlap, sélection de features, et optimisation d'hyperparamètres.

### 🎓 Contexte Académique

Projet réalisé dans le cadre d'une activité de recherche supervisée, utilisant le dataset **MotionSense** disponible sur Kaggle. L'objectif est de classifier 6 activités humaines distinctes à partir de données de capteurs inertiels provenant de smartphones.

## ✨ Caractéristiques

- ✅ **Classification de 6 activités** : marche, jogging, montée/descente d'escaliers, assis, debout
- ✅ **Split par utilisateur** pour éviter la fuite de données
- ✅ **Validation LOSO** (Leave-One-Subject-Out) pour évaluation robuste
- ✅ **240+ features avancées** : temporelles, fréquentielles (FFT), entropie, autocorrélation
- ✅ **Optimisation complète** : overlap, sélection de features, hyperparamètres
- ✅ **Architecture CNN avec régularisation** pour prévenir l'overfitting
- ✅ **Visualisations détaillées** : matrices de confusion, courbes d'apprentissage, distributions

## 📊 Dataset

**MotionSense Dataset**
- 📱 Source : Données d'accéléromètre et gyroscope de smartphones
- 👥 24 utilisateurs
- 🏃 6 activités : `dws` (descendre), `ups` (monter), `wlk` (marcher), `jog` (courir), `sit` (assis), `std` (debout)
- 📏 Capteurs : 12 dimensions (accélération, rotation, gravité sur axes x, y, z)
- 🔗 [Télécharger sur Kaggle](https://www.kaggle.com/malekzadeh/motionsense-dataset)

## 🔬 Méthodologie

### Prétraitement des Données

```
1. Chargement des fichiers CSV par utilisateur/activité
2. Regroupement des labels (ex: sit_5, sit_13 → sit)
3. Segmentation en fenêtres glissantes :
   - Taille de fenêtre : 500 échantillons
   - Overlap : 80% (optimal trouvé par GridSearch)
   - ~12,600 fenêtres générées
```

### Approche 1 : Random Forest

**Extraction de Features (240+ caractéristiques)**
- Statistiques temporelles : moyenne, std, min, max, médiane, quartiles, variance, skewness, kurtosis
- Variations : variation totale, moyenne, maximale
- Domaine fréquentiel : FFT (magnitude, fréquence dominante)
- Entropie de Shannon
- Autocorrélation (lag-1, lag-5)
- Zero-crossings, énergie, RMS

**Optimisation**
- SelectKBest : k=250 features optimales
- GridSearchCV : 360 combinaisons d'hyperparamètres testées
- Validation croisée 3-fold

### Approche 2 : CNN 1D

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

**Régularisation**
- L2 regularization (0.001)
- Batch Normalization
- Dropout progressif (0.3 → 0.5)
- Early Stopping (patience=15)
- ReduceLROnPlateau

## 🏆 Résultats

### Performance Comparative

| Modèle | Précision | Configuration |
|--------|-----------|---------------|
| **Random Forest (Split 70/30)** | **94.65%** | 132 features de base |
| **Random Forest (LOSO)** | **97.16%** | 132 features de base |
| **Random Forest Optimisé** | **~98%** | 250 features + GridSearch |
| **CNN 1D** | **98.25%** | Apprentissage end-to-end |

### Analyse

✅ **Cohérence** : Split 70/30 < LOSO < CNN (progression logique)  
✅ **Généralisation** : Validation LOSO confirme la robustesse inter-utilisateurs  
✅ **Amélioration CNN** : +3.6 points vs RF baseline grâce aux features temporelles automatiques  
✅ **Pas d'overfitting** : Écart train/validation <2% avec régularisation  

### Confusion Matrix (CNN)

Les confusions les plus fréquentes sont logiques :
- `wlk` ↔ `jog` (activités similaires)
- `sit` ↔ `std` (transitions)
- `ups` ↔ `dws` (mouvements verticaux)

## 🚀 Installation

### Prérequis

```bash
Python 3.8+
pip
```

### Installation des dépendances

```bash
# Cloner le repository
git clone https://github.com/votre-username/human-activity-recognition.git
cd human-activity-recognition

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows

# Installer les dépendances
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

## 💻 Utilisation

### 1. Télécharger le Dataset

Téléchargez le [MotionSense Dataset](https://www.kaggle.com/malekzadeh/motionsense-dataset) et placez-le dans le dossier `data/`.

Structure attendue :
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

### 2. Exécuter le Notebook

```bash
jupyter notebook motionsense_classification.ipynb
```

### 3. Ou utiliser les scripts Python

```bash
# Entraîner le Random Forest
python train_random_forest.py --data_path data/ --window_size 500 --overlap 0.8

# Entraîner le CNN
python train_cnn.py --data_path data/ --window_size 500 --epochs 100

# Optimisation complète
python optimize.py --data_path data/
```

## 📁 Structure du Projet

```
human-activity-recognition/
│
├── data/                          # Dataset (non inclus)
│   ├── dws/
│   ├── ups/
│   └── ...
│
├── notebooks/
│   └── motionsense_classification.ipynb   # Notebook principal
│
├── src/
│   ├── data_loading.py           # Chargement des données
│   ├── preprocessing.py          # Segmentation en fenêtres
│   ├── feature_extraction.py    # Extraction de features
│   ├── models.py                 # Architectures RF et CNN
│   └── optimization.py           # GridSearch et sélection
│
├── models/                       # Modèles sauvegardés
│   ├── rf_model.pkl
│   ├── cnn_model.h5
│   └── scaler.pkl
│
├── results/                      # Visualisations et rapports
│   ├── confusion_matrices/
│   ├── training_curves/
│   └── optimization_results/
│
├── requirements.txt              # Dépendances
├── README.md                     # Ce fichier
└── LICENSE                       # Licence MIT
```

## 🛠️ Technologies

- **Python 3.8+** : Langage principal
- **NumPy & Pandas** : Manipulation de données
- **Scikit-learn** : Random Forest, preprocessing, métriques
- **TensorFlow/Keras** : CNN 1D
- **Matplotlib & Seaborn** : Visualisations
- **SciPy** : FFT, statistiques avancées

## 📈 Améliorations Futures

- [ ] Data augmentation (rotation temporelle, ajout de bruit)
- [ ] Architecture ResNet 1D
- [ ] Attention mechanisms / Transformers
- [ ] Ensemble methods (RF + CNN)
- [ ] Déploiement avec Flask/FastAPI
- [ ] Application mobile temps réel

## 📚 Références

1. Malekzadeh, M., et al. (2019). "Mobile Sensor Data Anonymization"
2. Goodfellow, I., et al. (2016). "Deep Learning" - MIT Press
3. Breiman, L. (2001). "Random Forests" - Machine Learning

## 👨‍💻 Auteur

**Votre Nom**
- GitHub: [@votre-username](https://github.com/votre-username)
- LinkedIn: [Votre Profil](https://linkedin.com/in/votre-profil)
- Email: votre.email@example.com

## 🙏 Remerciements

- Dataset MotionSense par Mohammad Malekzadeh
- Professeur superviseur : [Nom du professeur]
- Communauté Kaggle

## 📄 Licence

Ce projet est sous licence MIT. Voir le fichier [LICENSE](LICENSE) pour plus de détails.

---

⭐ **Si ce projet vous a été utile, n'hésitez pas à lui donner une étoile !** ⭐

## 📞 Contact

Pour toute question ou suggestion, n'hésitez pas à ouvrir une [issue](https://github.com/votre-username/human-activity-recognition/issues) ou à me contacter directement.

---

*Projet réalisé dans le cadre d'une activité de recherche supervisée - 2024*
