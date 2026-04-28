# Fraud Detection — Enterprise ML Pipeline

Pipeline de détection de fraude bancaire de bout en bout, construit avec une rigueur niveau production. Ce document explique les choix effectués à chaque étape, les résultats obtenus et ce qu'ils signifient concrètement.

---

## Le problème traité

On cherche à prédire si une transaction bancaire est frauduleuse (`Fraudulent = 1`) ou non (`Fraudulent = 0`). C'est un problème de **classification binaire supervisée** avec un fort déséquilibre de classes : seulement **4.92 %** des transactions sont des fraudes, soit un ratio de **19:1** entre les deux classes.

Ce déséquilibre a des conséquences directes sur la modélisation : un modèle qui prédit "non fraude" à chaque fois atteindrait 95 % d'accuracy sans rien apprendre. C'est pourquoi l'accuracy est inutile ici, et pourquoi on privilégie le **recall** (ne pas rater de fraudes) et la **PR-AUC** (courbe précision-rappel, adaptée aux classes rares).

---

## Les données

Le dataset contient **51 000 transactions** décrites par 10 variables. Avant toute modélisation, **881 doublons exacts** ont été supprimés, ramenant le dataset à **50 119 observations nettes**.

**Cinq variables ont des valeurs manquantes**, toutes autour de 5 % :

| Variable | Manquants |
|---|---|
| Transaction_Amount | 4.94 % |
| Time_of_Transaction | 5.00 % |
| Device_Used | 4.86 % |
| Location | 4.99 % |
| Payment_Method | 4.84 % |

Ce taux uniforme de ~5 % sur plusieurs colonnes suggère un mécanisme de données manquantes aléatoire (MCAR), ce qui autorise une imputation par médiane ou mode sans biais majeur.

**Deux variables ont été supprimées** avant toute modélisation : `Transaction_ID` (identifiant unique non prédictif) et `User_ID` (identifiant à forte cardinalité qui risquerait de mémoriser des comportements individuels sans généraliser). Ces suppressions sont des décisions délibérées de prévention du data leakage.

---

## Ce que l'EDA a révélé

### Distributions numériques

Les montants (`Transaction_Amount`) varient de **5 € à 49 998 €**, avec une médiane à **2 524 €** et une moyenne à 2 999 €. La distribution est étalée vers la droite mais seulement **1 % d'outliers** selon la règle IQR — aucun nettoyage agressif n'est nécessaire.

Le nombre de transactions dans les 24 heures (`Number_of_Transactions_Last_24H`) va de 1 à 14, avec une médiane à 7. Distribution régulière, zéro outlier.

### Absence de signal prédictif

C'est le constat central de l'EDA. Les corrélations entre chaque feature numérique et la variable cible sont toutes inférieures à **0.006** en valeur absolue :

| Feature | Corrélation avec Fraudulent |
|---|---|
| Time_of_Transaction | +0.0059 |
| Transaction_Amount | +0.0058 |
| Account_Age | +0.0055 |
| Previous_Fraudulent_Transactions | +0.0008 |
| Number_of_Transactions_Last_24H | −0.0040 |

Des corrélations aussi proches de zéro indiquent que les features numériques ne permettent pas, individuellement, de distinguer les fraudes des transactions légitimes.

### Variables catégorielles : taux de fraude identiques

L'analyse bivariée des variables catégorielles confirme ce constat. Le taux de fraude est quasi-identique pour toutes les modalités :

- Par **type de transaction** : entre 4.67 % (ATM Withdrawal) et 5.14 % (Online Purchase)
- Par **appareil utilisé** : entre 4.66 % (Tablet) et 5.16 % (Mobile)
- Par **méthode de paiement** : entre 4.78 % et 5.11 %

Ces écarts sont statistiquement non significatifs sur 50 000 observations. Les variables catégorielles n'apportent donc pas plus de signal que les numériques.

---

## Prévention du data leakage

Le data leakage est la source la plus fréquente de performances artificiellement gonflées en machine learning. Le notebook prend des mesures explicites à chaque étape susceptible d'en causer.

**Le split train/test est effectué avant toute transformation apprenante.** Aucun paramètre (médiane, mode, encodage) n'est calculé sur l'ensemble du dataset — toutes les statistiques d'imputation sont apprises uniquement sur le train, puis appliquées mécaniquement au test.

**Le SMOTE est encapsulé dans le pipeline imblearn**, ce qui garantit qu'il ne s'applique jamais sur les folds de validation ni sur le test. Appliquer SMOTE avant le split est une erreur classique qui fait fuiter des observations synthétiques dans le jeu de validation et gonfle artificiellement le recall.

**Les doublons sont supprimés avant le split** pour éviter qu'une même observation se retrouve à la fois en entraînement et en test, ce qui gonflerait les métriques de généralisation.

**Les identifiants sont supprimés** pour éviter la mémorisation d'individus spécifiques sans généralisation hors du dataset.

---

## Feature engineering

Cinq nouvelles variables sont créées à partir des variables brutes, avec une justification métier pour chacune. Ce feature engineering est encapsulé dans un `FraudFeatureEngineer` — une classe sklearn personnalisée (`BaseEstimator`, `TransformerMixin`) — intégrée au pipeline, garantissant qu'aucune information du test ne fuite vers le train.

**`Is_Night`** — vaut 1 si la transaction a lieu entre 22h et 6h. Hypothèse : les fraudes sont surreprésentées la nuit, quand les systèmes de surveillance humaine sont réduits et les délais de réaction plus longs.

**`Is_Business_Hours`** — vaut 1 entre 9h et 18h. Complément du précédent, pour capturer les transactions hors heures ouvrées qui peuvent signaler un comportement atypique.

**`Has_Previous_Fraud`** — vaut 1 si le compte a déjà eu au moins une fraude. Transforme une variable continue en signal binaire plus direct : l'existence d'un antécédent est métier-pertinente indépendamment du nombre exact.

**`Amount_per_24H_Transaction`** — montant divisé par (nombre de transactions dans les 24H + 1). Capture la notion de transaction "anormalement grosse" par rapport à l'activité récente du compte. Le +1 évite une division par zéro.

**`Amount_to_Account_Age`** — montant divisé par (ancienneté du compte en mois + 1). Un gros montant sur un compte très récent est plus suspect que le même montant sur un compte de 10 ans.

---

## Pipeline de preprocessing

Après le feature engineering, on obtient **9 variables numériques** et **4 catégorielles**. Le preprocessing est entièrement géré par un `ColumnTransformer` sklearn :

- **Variables numériques** : imputation par médiane (robuste aux outliers), puis `StandardScaler` (centrage-réduction nécessaire pour la régression logistique)
- **Variables catégorielles** : imputation par le mode, puis `OneHotEncoding` avec `handle_unknown='ignore'` pour tolérer de nouvelles modalités inconnues en production

### Gestion du déséquilibre avec SMOTE

Plutôt que de simplement ajuster `class_weight`, le pipeline utilise **SMOTE** (Synthetic Minority Over-sampling Technique). SMOTE génère des observations synthétiques de la classe minoritaire (fraude) par interpolation entre observations réelles voisines dans l'espace des features, jusqu'à équilibrer les classes.

Concrètement : avant SMOTE, le train contient ~38 100 transactions légitimes et ~1 974 fraudes. Après SMOTE, les deux classes sont à parité, ce qui permet au modèle d'apprendre sans pencher systématiquement vers la classe majoritaire.

---

## Comparaison des modèles

Trois modèles sont comparés en **validation croisée stratifiée à 5 folds** sur l'ensemble d'entraînement. La stratification garantit que chaque fold respecte le ratio 95/5 de la distribution originale, évitant les folds déséquilibrés par hasard.

| Modèle | Rôle dans la comparaison |
|---|---|
| Logistic Regression | Baseline linéaire, interprétable, rapide à entraîner |
| Random Forest | Modèle à forte capacité, capte les interactions non linéaires |
| Decision Tree | Arbre simple, interprétable, intermédiaire en capacité |

Les trois modèles convergent vers des **performances équivalentes à un classifieur aléatoire** (ROC-AUC ≈ 0.50, PR-AUC ≈ 0.05), ce qui est cohérent avec l'absence de signal prédictif constatée dans l'EDA. La PR-AUC d'un classifieur aléatoire sur ce dataset est égale au taux de fraude (~0.049) — aucun modèle ne dépasse ce seuil de manière significative.

### Diagnostic surapprentissage vs sous-apprentissage

La comparaison des scores train vs validation révèle le comportement de chaque modèle :

- **Random Forest** : scores train élevés, scores validation proches du hasard → **surapprentissage marqué**. Le modèle mémorise le bruit du train sans apprendre de signal généralisable.
- **Logistic Regression** : scores train et validation identiquement faibles → **sous-apprentissage**. La relation entre features et cible n'est pas linéaire, ou n'existe tout simplement pas.
- **Decision Tree** : comportement intermédiaire selon la profondeur, mais même constat final — pas de signal à apprendre.

---

## Optimisation du Decision Tree

Un `RandomizedSearchCV` est appliqué sur le Decision Tree avec **8 itérations** sur l'espace suivant :

```
max_depth        : [None, 5, 10, 15, 20]
min_samples_split: [2, 5, 10]
min_samples_leaf : [1, 2, 4]
smote__k_neighbors: [3, 5, 7]
```

La métrique d'optimisation est la **PR-AUC** (`average_precision`), plus discriminante que la ROC-AUC en présence de fort déséquilibre. L'optimisation est réalisée entièrement sur le train — le test n'est jamais touché pendant cette phase.

> Note : avec 5 × 3 × 3 × 3 = 135 combinaisons possibles, 8 itérations n'explorent que 6 % de l'espace. En pratique, passer à 20-30 itérations donnerait une optimisation plus robuste.

---

## Résultats finaux sur le test

L'évaluation finale est réalisée **une seule fois** sur le jeu de test hold-out (20 % du dataset, soit ~10 024 observations). C'est la règle fondamentale de toute évaluation rigoureuse : le test ne sert qu'à cette mesure finale, jamais pendant le développement ni la sélection de modèle.

| Métrique | Valeur obtenue |
|---|---|
| ROC-AUC | ~0.50 |
| PR-AUC | ~0.05 |
| Recall (fraude) | Variable selon le seuil |
| Precision (fraude) | Variable selon le seuil |

La cohérence entre scores de validation croisée et scores de test confirme l'absence de data leakage résiduel. Le modèle ne performe pas bien, mais il **performe honnêtement**.

---

## Interprétabilité

### Importances de variables (Decision Tree)

Les importances du Decision Tree placent `Transaction_Amount`, `Time_of_Transaction` et `Number_of_Transactions_Last_24H` en tête. Ce résultat reflète la structure du dataset — ces variables ont la plus grande variance et permettent les splits les plus nets au sens de l'impureté de Gini — et non un vrai pouvoir prédictif sur la fraude.

### Coefficients de la Logistic Regression

Entraînée en parallèle à des fins d'interprétabilité, la régression logistique produit des coefficients tous proches de zéro, confirmant l'absence de relation linéaire exploitable entre les features et la cible.

### Permutation importance

La permutation importance, calculée sur le test, mesure la dégradation de la PR-AUC lorsqu'on mélange aléatoirement les valeurs d'une feature. Des valeurs proches de zéro — attendues ici — indiquent que retirer n'importe quelle feature ne change pas les performances, ce qui confirme l'absence de signal.

---

## Ce que les résultats signifient vraiment

Un ROC-AUC de 0.50 et une PR-AUC de 0.05 ne signifient pas que le pipeline est mal construit. Ils signifient que **le dataset ne contient pas d'information suffisante pour distinguer les fraudes des transactions légitimes** avec les variables disponibles.

C'est un résultat précieux : un pipeline avec data leakage aurait affiché des ROC-AUC de 0.85-0.95 en donnant l'illusion de performances solides, pour s'effondrer en production. Ici, 0.50 en validation croisée et 0.50 en test — la cohérence est le signe d'une évaluation rigoureuse.

---

## Limites identifiées

**Dataset synthétique.** Les taux de fraude par modalité catégorielle sont trop uniformes (4.66 % à 5.16 %), et les corrélations numériques sont toutes inférieures à 0.006. Un dataset réel de fraude bancaire présente des patterns bien plus marqués — par exemple, des géographies ou des horaires surreprésentés.

**Valeurs manquantes uniformes.** 5 % sur exactement 5 colonnes indépendantes a un caractère artificiel. En production, les données manquantes ont des causes spécifiques et non aléatoires qui portent elles-mêmes un signal.

**Pas de dimension temporelle.** Les systèmes de détection de fraude performants exploitent des séquences d'événements par compte (vélocité, anomalie de comportement récent), pas des transactions isolées.

**Seuil de décision non calibré.** Le seuil par défaut (0.5) n'est pas optimal pour la détection de fraude. En pratique, le seuil doit être calibré selon le ratio coût(faux négatif) / coût(faux positif) propre au contexte métier.

---

## Pour aller plus loin en production réelle

| Axe | Action recommandée |
|---|---|
| Données | Enrichir avec vélocité, géolocalisation, device fingerprint, historique client agrégé |
| Seuil | Calibrer selon le ratio coût métier FN / FP |
| Monitoring | Suivre le drift des scores et du taux de fraude réel observé |
| Optimisation | Passer `n_iter` à 20-30 dans le `RandomizedSearchCV` |
| Interprétabilité | Ajouter SHAP pour des explications par transaction individuelle |
| Métriques opérationnelles | Suivre séparément recall fraude, precision fraude et taux d'alertes générées |

---

*Projet réalisé dans le cadre d'un cours de Data mining — Semestre 6.*
