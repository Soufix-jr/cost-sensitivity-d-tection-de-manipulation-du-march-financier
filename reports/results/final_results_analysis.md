# Analyse finale des résultats

## 1. Rappel de l'objectif

L'objectif du projet est de détecter les manipulations de marché en minimisant le coût total des erreurs. Dans ce contexte, toutes les erreurs n'ont pas la même gravité.

La fonction de coût utilisée est :

```text
Coût total = c_FN × FN + c_FP × FP
```

avec :

```text
c_FN = 15 000 000 €
c_FP = 7 000 €
```

Un faux négatif correspond à une manipulation réelle non détectée. C'est l'erreur la plus coûteuse. Un faux positif correspond à une fausse alerte, dont le coût est limité au traitement de l'enquête.

---

## 2. Résultat global

Le meilleur modèle en coût total sur le test set est :

```text
random_forest_smote — SMOTE + Threshold 0.5
```

Ses résultats sont :

| Métrique | Valeur |
|---|---:|
| Accuracy | 0.9919 |
| Precision | 0.7143 |
| Recall | 1.0000 |
| F1-score | 0.8333 |
| ROC-AUC | 0.9977 |
| Faux positifs | 10 |
| Faux négatifs | 0 |
| Vrais positifs | 25 |
| Vrais négatifs | 1194 |
| Coût total | 70 000 € |

Ce résultat signifie que le modèle détecte toutes les manipulations du test set, au prix de quelques fausses alertes. Comme le coût d'un faux positif est faible par rapport au coût d'un faux négatif, cette stratégie est économiquement très favorable.

---

## 3. Comparaison entre SMOTE et Threshold Moving

### 3.1 Meilleur modèle SMOTE

| Élément | Valeur |
|---|---:|
| Modèle | random_forest_smote |
| Méthode | SMOTE + Threshold 0.5 |
| Coût | 70 000 € |
| FN | 0 |
| FP | 10 |
| Recall | 1.0000 |
| Precision | 0.7143 |

### 3.2 Meilleur modèle Threshold Moving

| Élément | Valeur |
|---|---:|
| Modèle | xgboost_calibrated_isotonic |
| Méthode | Calibration + Threshold Moving |
| Seuil optimal | 0.135786 |
| Coût | 15 021 000 € |
| FN | 1 |
| FP | 3 |
| Recall | 0.9600 |
| Precision | 0.8889 |

Dans cette expérimentation, SMOTE obtient le coût le plus faible, principalement parce qu'il ne rate aucune manipulation. Le meilleur modèle Threshold Moving rate une manipulation, ce qui entraîne immédiatement un coût très élevé à cause de la valeur de c_FN.

---

## 4. Interprétation critique

Le résultat ne signifie pas que SMOTE est toujours supérieur. Il signifie que, sur ce test set précis, la stratégie SMOTE + Random Forest a permis d'obtenir zéro faux négatif.

Cependant, SMOTE possède des limites importantes :

- il modifie artificiellement la distribution d'apprentissage ;
- il crée des observations synthétiques qui peuvent être peu réalistes économiquement ;
- il ne tient pas directement compte de la matrice de coûts ;
- il optimise un problème statistique, pas directement un problème de décision métier.

Threshold Moving reste plus propre du point de vue décisionnel, car le seuil est explicitement relié aux coûts économiques des erreurs. Cette approche est donc plus facile à justifier dans un cadre réglementaire ou devant un analyste métier.

---

## 5. Tableau final de comparaison

| Rang | Modèle | Méthode | Seuil | Precision | Recall | F1-score | Coût | FP | FN | TP | TN |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | random_forest_smote | SMOTE + Threshold 0.5 | 0.500000 | 0.7143 | 1.0000 | 0.8333 | 70 000 € | 10 | 0 | 25 | 1194 |
| 2 | xgboost_calibrated_isotonic | Calibration + Threshold Moving | 0.135786 | 0.8889 | 0.9600 | 0.9231 | 15 021 000 € | 3 | 1 | 24 | 1201 |
| 3 | random_forest_calibrated_sigmoid | Calibration + Threshold Moving | 0.040396 | 0.7059 | 0.9600 | 0.8136 | 15 070 000 € | 10 | 1 | 24 | 1194 |
| 4 | random_forest_calibrated_sigmoid | Recall-Constrained Threshold | 0.065295 | 0.7059 | 0.9600 | 0.8136 | 15 070 000 € | 10 | 1 | 24 | 1194 |
| 5 | xgboost_calibrated_isotonic | Recall-Constrained Threshold | 0.400000 | 0.8846 | 0.9200 | 0.9020 | 30 021 000 € | 3 | 2 | 23 | 1201 |
| 6 | logistic_regression_calibrated_sigmoid | Recall-Constrained Threshold | 0.215668 | 0.7419 | 0.9200 | 0.8214 | 30 056 000 € | 8 | 2 | 23 | 1196 |
| 7 | logistic_regression_calibrated_sigmoid | Calibration + Threshold Moving | 0.202880 | 0.7188 | 0.9200 | 0.8070 | 30 063 000 € | 9 | 2 | 23 | 1195 |

---

## 6. Conclusion finale

La meilleure performance empirique est obtenue avec Random Forest + SMOTE, car cette méthode atteint un rappel parfait sur le test set et évite totalement les faux négatifs.

Cependant, la méthode centrale du projet reste Calibration + Threshold Moving, car elle formalise directement la décision à partir de la théorie de Bayes et de la matrice de coûts. Elle permet de transformer un modèle probabiliste en outil de décision économique.

La conclusion correcte à présenter est donc :

```text
SMOTE domine empiriquement sur ce test set.
Threshold Moving reste plus justifiable théoriquement, économiquement et réglementairement.
La comparaison montre l'écart entre performance observée et robustesse décisionnelle.
```
