#  Fake News Detection with DistilBERT

> Projet LLM — Détection automatique de fake news par fine-tuning de modèles de langage pré-entraînés.

---

##  Problématique

La désinformation en ligne est un problème majeur. Ce projet propose un système de **détection automatique de fake news** en utilisant des modèles de langage modernes (DistilBERT) fine-tunés sur un dataset réel d'articles de presse.

**Question centrale :** Dans quelle mesure un modèle Transformer fine-tuné peut-il dépasser les approches classiques TF-IDF pour distinguer les articles fiables des articles trompeurs ?

---


## Dataset

**Kaggle Fake News Dataset** — Clément Bisaillon  
 [https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)

Le dataset contient :
- `True.csv` : ~21 000 articles réels (source : Reuters)
- `Fake.csv` : ~23 000 articles fictifs/trompeurs

**Labels :**
- `0` → Fake news
- `1` → Real news

##  Utilisation

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

##  Modèle

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


