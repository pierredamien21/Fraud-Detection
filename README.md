# Fraud Detection — Decision Tree Classifier

Détection de transactions frauduleuses à l'aide d'un pipeline de Machine Learning complet, basé sur un **Decision Tree Classifier** (scikit-learn).

---

## Description du projet

Ce projet vise à identifier automatiquement les transactions bancaires frauduleuses à partir de données comportementales et transactionnelles. Il couvre l'intégralité du workflow ML : exploration des données, feature engineering, transformations statistiques, réduction dimensionnelle, entraînement, optimisation et évaluation du modèle.

Le modèle utilisé est exclusivement le **Decision Tree Classifier**, choisi pour sa capacité d'interprétation et sa transparence dans un contexte de détection de fraude.

---

## Structure du dépôt

```
fraud-detection-decision-tree/
│
├── Fraud_Detection_Dataset.csv          # Dataset source (51 000 transactions)
├── Fraud_Detection_DecisionTree.ipynb   # Notebook principal (pipeline complet)
└── README.md                            # Ce fichier
```

---

## Dataset

| Attribut | Valeur |
|---|---|
| Nombre de transactions | 51 000 |
| Nombre de features | 12 |
| Variable cible | `Fraudulent` (0 = légitime, 1 = fraude) |
| Taux de fraude | ~4.9% (dataset déséquilibré) |

**Features disponibles :**

- `Transaction_ID`, `User_ID` — identifiants (supprimés avant modélisation)
- `Transaction_Amount` — montant de la transaction
- `Transaction_Type` — type (ATM Withdrawal, Online Purchase, etc.)
- `Time_of_Transaction` — heure de la transaction
- `Device_Used` — appareil utilisé (Mobile, Desktop, Tablet)
- `Location` — ville de la transaction
- `Previous_Fraudulent_Transactions` — historique de fraude de l'utilisateur
- `Account_Age` — ancienneté du compte (jours)
- `Number_of_Transactions_Last_24H` — activité récente
- `Payment_Method` — moyen de paiement
- `Fraudulent` — variable cible

---

## Pipeline ML — Étapes détaillées

### 1. Analyse Exploratoire (EDA)
- Visualisation des valeurs manquantes (~5% par colonne concernée)
- Distribution de la variable cible : fort déséquilibre (95% légitime / 5% fraude)
- Histogrammes des variables numériques par classe
- Analyse des variables catégorielles (`Transaction_Type`, `Device_Used`, `Location`, `Payment_Method`)
- Matrice de corrélation
- Boxplots des montants et de l'activité par classe

### 2. Feature Engineering
- Suppression des identifiants non informatifs (`Transaction_ID`, `User_ID`)
- Imputation des valeurs manquantes :
  - Variables numériques → médiane (robuste aux outliers)
  - Variables catégorielles → mode
- Nouvelles features créées :
  - `Time_Slot` : tranche horaire (morning / afternoon / evening / night)
  - `Fraud_Risk_Ratio` : ratio fraudes passées / ancienneté du compte
  - `High_Activity` : flag binaire si l'utilisateur dépasse le 75e percentile d'activité en 24h
  - `High_Amount` : flag binaire si le montant dépasse le 75e percentile

### 3. Transformations statistiques
- **Log-transform** (`log1p`) sur `Transaction_Amount` : correction de l'asymétrie de distribution
- **One-Hot Encoding** (drop_first) sur les 5 variables catégorielles
- **StandardScaler** : normalisation avant l'ACP

### 4. Rééquilibrage des classes
- Méthode : oversampling de la classe minoritaire (fraude) par rééchantillonnage aléatoire avec remplacement (`sklearn.utils.resample`)
- Résultat : ~48 490 observations par classe

### 5. Réduction dimensionnelle — ACP
- Analyse en Composantes Principales (PCA) appliquée après normalisation
- Critère de sélection : **95% de la variance expliquée**
- Résultat : réduction de ~30 dimensions vers **24 composantes principales**
- Visualisation : scree plot, courbe de variance cumulée, projection 2D (PC1 vs PC2)

### 6. Entraînement du modèle
- Modèle : `DecisionTreeClassifier` (scikit-learn)
- Split Train/Test : 80% / 20% stratifié
- Modèle baseline entraîné sans tuning pour établir une référence

### 7. Cross-Validation
- StratifiedKFold — 5 folds, shuffle activé
- Métriques évaluées : Accuracy, Precision, Recall, F1-Score, ROC-AUC
- Résultats visualisés sous forme de boxplots

### 8. GridSearchCV — Optimisation des hyperparamètres
- 240 combinaisons testées (x 5 folds = 1 200 fits)
- Grille de recherche :

| Hyperparamètre | Valeurs testées |
|---|---|
| `criterion` | `gini`, `entropy` |
| `max_depth` | `3`, `5`, `10`, `15`, `None` |
| `min_samples_split` | `2`, `5`, `10`, `20` |
| `min_samples_leaf` | `1`, `5`, `10` |
| `class_weight` | `None`, `balanced` |

- Métrique d'optimisation : F1-Score (adapté aux classes déséquilibrées)

### 9. Evaluation du modèle final
- Rapport de classification complet (precision, recall, F1 par classe)
- Matrice de confusion annotée (TP, TN, FP, FN)
- Courbe ROC avec AUC
- Courbe Précision-Rappel
- Importance des features (Gini impurity)
- Visualisation de l'arbre de décision (profondeur limitée à 4 pour la lisibilité)
- Comparaison Baseline vs modèle optimisé

---

## Résultats

| Métrique | Baseline | Modèle optimisé (GridSearch) |
|---|---|---|
| Accuracy | — | ~97% |
| F1-Score | — | ~97% |
| ROC-AUC | — | ~97% |

Les valeurs exactes sont affichées dans le notebook après exécution complète.

---

## Installation & Exécution

### Prérequis

```bash
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

### Lancer le notebook

```bash
git clone https://github.com/<votre-username>/fraud-detection-decision-tree.git
cd fraud-detection-decision-tree
jupyter notebook Fraud_Detection_DecisionTree.ipynb
```

> Assurez-vous que `Fraud_Detection_Dataset.csv` est dans le même répertoire que le notebook.

---

## Technologies utilisées

| Outil | Usage |
|---|---|
| Python 3.x | Langage principal |
| pandas | Manipulation des données |
| numpy | Calcul numérique |
| matplotlib / seaborn | Visualisations |
| scikit-learn | ML pipeline complet |
| Jupyter Notebook | Environnement d'expérimentation |

---

## Points clés du projet

- Pipeline ML de bout en bout documenté et reproductible
- Gestion rigoureuse du déséquilibre de classes (oversampling)
- Transformations statistiques justifiées (log-transform, ACP)
- Optimisation systématique des hyperparamètres (GridSearchCV)
- Evaluation multi-métriques adaptée à la détection de fraude
- Modèle interprétable et visualisable (arbre de décision)

---

## Auteur

Projet réalisé dans le cadre d'un apprentissage du Machine Learning appliqué à la détection de fraude bancaire.
