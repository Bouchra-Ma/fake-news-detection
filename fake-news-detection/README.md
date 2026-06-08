# 🕵️ Fake News Detection with DistilBERT

> Projet LLM — Détection automatique de fake news par fine-tuning de modèles de langage pré-entraînés.

---

## 📌 Problématique

La désinformation en ligne est un problème majeur. Ce projet propose un système de **détection automatique de fake news** en utilisant des modèles de langage modernes (DistilBERT) fine-tunés sur un dataset réel d'articles de presse.

**Question centrale :** Dans quelle mesure un modèle Transformer fine-tuné peut-il dépasser les approches classiques TF-IDF pour distinguer les articles fiables des articles trompeurs ?

---

## 📁 Structure du projet

```
fake-news-detection/
│
├── data/                          # Données (à télécharger, voir ci-dessous)
│   ├── True.csv                   # Articles réels
│   └── Fake.csv                   # Articles fictifs/faux
│
├── notebooks/
│   └── fake_news_detection_demo.ipynb   # Notebook de démonstration complet
│
├── src/
│   ├── explore_data.py            # EDA & prétraitement
│   ├── baselines.py               # Modèles classiques (TF-IDF + LR/NB/SVM)
│   ├── train.py                   # Fine-tuning DistilBERT
│   └── predict.py                 # Inférence sur de nouveaux textes
│
├── outputs/                       # Générés automatiquement après entraînement
│   ├── best_model/                # Modèle sauvegardé
│   ├── training_history.png
│   ├── confusion_matrix.png
│   └── metrics.csv
│
├── requirements.txt
└── README.md
```

---

## 📚 Dataset

**Kaggle Fake News Dataset** — Clément Bisaillon  
👉 [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Le dataset contient :
- `True.csv` : ~21 000 articles réels (source : Reuters)
- `Fake.csv` : ~23 000 articles fictifs/trompeurs

**Labels :**
- `0` → Fake news
- `1` → Real news

> ⚠️ Si les fichiers CSV ne sont pas présents dans `data/`, les scripts génèrent automatiquement des données synthétiques de démonstration.

### Téléchargement

```bash
# Option 1 : Kaggle CLI
pip install kaggle
kaggle datasets download -d clmentbisaillon/fake-and-real-news-dataset
unzip fake-and-real-news-dataset.zip -d data/

# Option 2 : Manuellement
# Télécharger True.csv et Fake.csv depuis Kaggle et les placer dans data/
```

---

## 🚀 Installation

### Prérequis
- Python 3.9+
- GPU recommandé (CUDA) mais CPU fonctionne

### Installation des dépendances

```bash
git clone https://github.com/YOUR_USERNAME/fake-news-detection.git
cd fake-news-detection
pip install -r requirements.txt
```

---

## ▶️ Utilisation

### 1. Exploration des données (EDA)

```bash
python src/explore_data.py
```
Génère des visualisations dans `outputs/eda/` :
- Distribution des classes
- Distribution du nombre de mots
- Nuages de mots (wordclouds)

### 2. Modèles de référence (Baselines)

```bash
python src/baselines.py
```
Entraîne et évalue :
- Logistic Regression + TF-IDF
- Naive Bayes + TF-IDF
- Linear SVM + TF-IDF

### 3. Fine-tuning DistilBERT

```bash
python src/train.py
```

- Entraîne DistilBERT sur le dataset
- Sauvegarde le meilleur modèle dans `outputs/best_model/`
- Génère les courbes d'entraînement et la matrice de confusion

### 4. Prédiction sur de nouveaux articles

```bash
# Mode interactif
python src/predict.py

# Texte direct
python src/predict.py --text "Scientists confirm new vaccine is 95% effective."

# Depuis un fichier
python src/predict.py --file article.txt
```

### 5. Notebook de démonstration

```bash
jupyter notebook notebooks/fake_news_detection_demo.ipynb
```

---

## 🧪 Modèle

| Composant | Détail |
|-----------|--------|
| Architecture | DistilBERT (distilbert-base-uncased) |
| Paramètres | ~66M |
| Max longueur | 256 tokens |
| Batch size | 16 |
| Epochs | 3 |
| Learning rate | 2e-5 (avec warmup) |
| Optimiseur | AdamW |

---

## 📊 Résultats

### Métriques sur le jeu de test

| Modèle | Accuracy | F1-score (weighted) |
|--------|----------|---------------------|
| Naive Bayes (TF-IDF) | ~0.934 | ~0.934 |
| Logistic Regression (TF-IDF) | ~0.985 | ~0.985 |
| Linear SVM (TF-IDF) | ~0.992 | ~0.992 |
| **DistilBERT (fine-tuné)** | **~0.994** | **~0.994** |

> Les résultats exacts dépendent de la seed et de l'infrastructure d'entraînement.

### Visualisations

Après l'entraînement, les fichiers suivants sont disponibles dans `outputs/` :
- `training_history.png` — Courbes de loss, accuracy et F1 par epoch
- `confusion_matrix.png` — Matrice de confusion sur le test set
- `metrics.csv` — Métriques finales

---

## 🔬 Analyse

### Pourquoi DistilBERT ?

- **Contextualisé** : comprend le sens des mots selon leur contexte
- **Pré-entraîné** : bénéficie de la compréhension du langage naturel à grande échelle
- **Efficace** : 40% plus rapide que BERT avec 97% de ses performances
- **Adapté** : peut capturer des patterns stylistiques subtils (ton alarmiste, manque de sources...)

### Limites

- Le dataset Kaggle est assez facile → performances très élevées même avec des baselines
- Risque de biais de style (source des articles)
- Ne vérifie pas les faits réels (pas de base de connaissances)

### Pistes d'amélioration

- Tester sur **LIAR** dataset (6 classes de véracité, plus difficile)
- Utiliser **RoBERTa** ou **DeBERTa** pour de meilleures performances
- Système **RAG** : vérification des faits via recherche web en temps réel
- Intégrer les métadonnées (source, auteur, date)

---

## 🏗️ Reproductibilité

Toutes les graines aléatoires sont fixées (`seed=42`). Pour reproduire exactement :

```bash
PYTHONHASHSEED=42 python src/train.py
```

---

## 📄 Références

- Devlin et al. (2019). [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)
- Sanh et al. (2019). [DistilBERT, a distilled version of BERT](https://arxiv.org/abs/1910.01108)
- Ahmed et al. (2017). [Detection of Online Fake News Using N-Gram Analysis and Machine Learning Techniques](https://link.springer.com/chapter/10.1007/978-3-319-67071-3_11)
- Dataset : [Kaggle Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

---

## 📝 Licence

MIT License — Projet académique LLM.
