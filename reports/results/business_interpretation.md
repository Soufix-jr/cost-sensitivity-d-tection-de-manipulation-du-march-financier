# Interprétation business des résultats
## 1. Objectif du projet
L'objectif du projet est de détecter des situations suspectes de manipulation de marché en utilisant une approche de machine learning sensible aux coûts.
Contrairement à une classification classique, le but n'est pas seulement de maximiser l'accuracy. Le but principal est de minimiser le coût économique total des erreurs.
La fonction de coût utilisée est :
```text
Coût total = c_FN × FN + c_FP × FP
```
avec :
```text
c_FN = 15 000 000 €
c_FP = 7 000 €
```
Un faux négatif correspond à une manipulation non détectée. C'est l'erreur la plus grave, car elle peut entraîner une sanction réglementaire, une perte financière ou un risque de réputation.
Un faux positif correspond à une fausse alerte. Son coût est plus faible, car il représente principalement un coût d'enquête interne.
---
## 2. Seuil classique contre seuil cost-sensitive
Dans une classification classique, on utilise généralement un seuil de décision égal à 0.5.
```text
Si P(manipulation) >= 0.5 → alerte
Sinon → normal
```
Cette règle est insuffisante dans notre cas, car les coûts des erreurs sont très asymétriques.
Le seuil théorique de Bayes est :
```text
t* = c_FP / (c_FN + c_FP)
```
Avec les coûts utilisés dans ce projet :
```text
t* = 7000 / (15000000 + 7000)
t* ≈ 0.000466
```
Cela signifie qu'une alerte peut être déclenchée même avec une probabilité faible de manipulation, car le coût d'une manipulation ratée est extrêmement élevé.
---
## 3. Meilleur modèle obtenu
Le meilleur modèle selon le coût total final est :
```text
xgboost_calibrated_isotonic
```
Son seuil optimal appris sur l'ensemble de validation est :
```text
seuil optimal = 0.135786
```
Résultats sur le test set :
| Métrique | Valeur |
|---|---:|
| Precision | 0.8889 |
| Recall | 0.9600 |
| F1-score | 0.9231 |
| ROC-AUC | 0.9790 |
| Faux positifs | 3 |
| Faux négatifs | 1 |
| Vrais positifs | 24 |
| Vrais négatifs | 1201 |
| Coût total optimal | 15 021 000 € |
| Savings | 74.97 % |
---
## 4. Lecture métier des résultats
Le modèle final détecte 24 cas suspects sur les manipulations présentes dans le test set.
Il rate seulement 1 manipulation(s), ce qui est important dans un contexte réglementaire où le coût d'un faux négatif est très élevé.
Le modèle génère 3 fausse(s) alerte(s). Ce nombre reste acceptable, car le coût d'une enquête est faible par rapport au coût d'une manipulation non détectée.
La logique métier est donc :
```text
Il vaut mieux enquêter sur quelques faux positifs que rater une vraie manipulation.
```
---
## 5. Comparaison des modèles après Threshold Moving
| Modèle | Seuil optimal | Coût seuil 0.5 | Coût optimal | Savings | FN optimal | FP optimal |
|---|---:|---:|---:|---:|---:|---:|
| logistic_regression_calibrated_sigmoid | 0.202880 | 180 021 000 € | 30 063 000 € | 83.30 % | 2 | 9 |
| random_forest_calibrated_sigmoid | 0.040396 | 90 035 000 € | 15 070 000 € | 83.26 % | 1 | 10 |
| xgboost_calibrated_isotonic | 0.135786 | 60 014 000 € | 15 021 000 € | 74.97 % | 1 | 3 |
---
## 6. Analyse de la calibration
La calibration est nécessaire car le Threshold Moving suppose que les probabilités prédites sont interprétables comme de vraies probabilités.
| Modèle | Méthode de calibration | Brier score | ROC-AUC | Log loss |
|---|---|---:|---:|---:|
| logistic_regression | sigmoid | 0.007151 | 0.997209 | 0.024712 |
| random_forest | sigmoid | 0.005828 | 0.997907 | 0.019956 |
| xgboost | isotonic | 0.003601 | 0.978987 | 0.090982 |
---
## 7. Conclusion technique
Les résultats confirment que le déplacement du seuil permet de réduire fortement le coût total par rapport à un seuil classique de 0.5.
La démarche complète est :
```text
1. Entraîner un modèle probabiliste
2. Calibrer les probabilités
3. Chercher le seuil qui minimise le coût métier
4. Évaluer uniquement sur le test set final
```
Cette approche est plus adaptée qu'une simple maximisation de l'accuracy, car elle intègre directement l'asymétrie économique entre faux positifs et faux négatifs.
---
## 8. Conclusion business
Le modèle final fournit un outil d'aide à la surveillance de marché. Il ne remplace pas l'analyste humain. Il priorise les alertes selon leur risque estimé et leur impact économique potentiel.
L'intérêt principal du système est de transformer un modèle de classification classique en un outil de décision orienté coût, conforme à une logique de contrôle, de backtesting et de justification réglementaire.
