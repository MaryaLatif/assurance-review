# 🛡️ InsurReview AI — Analyse d'avis d'assurance par NLP

> Projet de Supervised Learning — NLP | ESILV A4 S8

Application Streamlit d'analyse d'avis clients d'assurance, basée sur un modèle **DistilBERT fine-tuné** sur 24 000+ avis.

---

## 📋 Fonctionnalités

| Page | Description |
|------|-------------|
| 🏠 Accueil | Description du projet et des modèles comparés |
| 🔮 Prediction | Prédiction de note (1–5 ⭐) avec DistilBERT |
| 📊 Summary | Statistiques et visualisations du dataset |
| 💡 Explanation | Mots influents (TF-IDF) + probabilités par classe |
| 🔍 Information Retrieval | Recherche d'avis similaires par cosine similarity |

---

## 🗂️ Structure du projet

```
assurance-review/
│
├── app.py                        # Application Streamlit
├── predict.py                    # Fonctions de prédiction DistilBERT
│
├── distilbert_review_model/      # Modèle fine-tuné (chargé via Git LFS)
│   ├── config.json
│   ├── model.safetensors
│   ├── tokenizer.json
│   └── tokenizer_config.json
│
├── datas/
│   ├── data_supervised.csv       # Dataset principal
│   └── …
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation et lancement

### Prérequis

- Python 3.10+
- Git LFS installé (pour récupérer le modèle)

### 1. Cloner le repo

```bash
git lfs install
git clone https://github.com/MaryaLatif/assurance-review.git
cd assurance-review
```

> ⚠️ Le `git lfs install` **avant** le clone est indispensable pour télécharger `model.safetensors` (255 MB).

### 2. Créer un environnement virtuel

```bash
python3 -m venv venv
source venv/bin/activate        # Mac/Linux
# ou
venv\Scripts\activate           # Windows
```

### 3. Installer les dépendances

```bash
pip install -r requirements.txt
```

### 4. Lancer l'application

```bash
streamlit run app.py
```

L'app s'ouvre automatiquement sur [http://localhost:8501](http://localhost:8501).
