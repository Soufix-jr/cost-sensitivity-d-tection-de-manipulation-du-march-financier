import pandas as pd
from pathlib import Path


RESULTS_DIR = Path("reports/results")
OUTPUT_FILE = RESULTS_DIR / "business_interpretation.md"

THRESHOLD_RESULTS_FILE = RESULTS_DIR / "threshold_optimization_results.csv"
CALIBRATION_RESULTS_FILE = RESULTS_DIR / "calibrated_models_results.csv"

C_FN = 15_000_000
C_FP = 7_000


def euros(x):
    return f"{x:,.0f} €".replace(",", " ")


def percent(x):
    return f"{x * 100:.2f} %"


def main():
    threshold_df = pd.read_csv(THRESHOLD_RESULTS_FILE)
    calibration_df = pd.read_csv(CALIBRATION_RESULTS_FILE)

    best_model = threshold_df.sort_values("cost_opt").iloc[0]

    bayes_threshold = C_FP / (C_FN + C_FP)

    lines = []

    lines.append("# Interprétation business des résultats\n")

    lines.append("## 1. Objectif du projet\n")
    lines.append(
        "L'objectif du projet est de détecter des situations suspectes de manipulation de marché "
        "en utilisant une approche de machine learning sensible aux coûts.\n"
    )
    lines.append(
        "Contrairement à une classification classique, le but n'est pas seulement de maximiser "
        "l'accuracy. Le but principal est de minimiser le coût économique total des erreurs.\n"
    )

    lines.append("La fonction de coût utilisée est :\n")
    lines.append("```text\n")
    lines.append("Coût total = c_FN × FN + c_FP × FP\n")
    lines.append("```\n")

    lines.append("avec :\n")
    lines.append("```text\n")
    lines.append(f"c_FN = {euros(C_FN)}\n")
    lines.append(f"c_FP = {euros(C_FP)}\n")
    lines.append("```\n")

    lines.append(
        "Un faux négatif correspond à une manipulation non détectée. C'est l'erreur la plus grave, "
        "car elle peut entraîner une sanction réglementaire, une perte financière ou un risque de réputation.\n"
    )
    lines.append(
        "Un faux positif correspond à une fausse alerte. Son coût est plus faible, car il représente "
        "principalement un coût d'enquête interne.\n"
    )

    lines.append("---\n")

    lines.append("## 2. Seuil classique contre seuil cost-sensitive\n")
    lines.append(
        "Dans une classification classique, on utilise généralement un seuil de décision égal à 0.5.\n"
    )
    lines.append("```text\n")
    lines.append("Si P(manipulation) >= 0.5 → alerte\n")
    lines.append("Sinon → normal\n")
    lines.append("```\n")

    lines.append(
        "Cette règle est insuffisante dans notre cas, car les coûts des erreurs sont très asymétriques.\n"
    )

    lines.append("Le seuil théorique de Bayes est :\n")
    lines.append("```text\n")
    lines.append("t* = c_FP / (c_FN + c_FP)\n")
    lines.append("```\n")

    lines.append("Avec les coûts utilisés dans ce projet :\n")
    lines.append("```text\n")
    lines.append(f"t* = {C_FP} / ({C_FN} + {C_FP})\n")
    lines.append(f"t* ≈ {bayes_threshold:.6f}\n")
    lines.append("```\n")

    lines.append(
        "Cela signifie qu'une alerte peut être déclenchée même avec une probabilité faible de manipulation, "
        "car le coût d'une manipulation ratée est extrêmement élevé.\n"
    )

    lines.append("---\n")

    lines.append("## 3. Meilleur modèle obtenu\n")

    lines.append("Le meilleur modèle selon le coût total final est :\n")
    lines.append("```text\n")
    lines.append(f"{best_model['model']}\n")
    lines.append("```\n")

    lines.append("Son seuil optimal appris sur l'ensemble de validation est :\n")
    lines.append("```text\n")
    lines.append(f"seuil optimal = {best_model['best_threshold']:.6f}\n")
    lines.append("```\n")

    lines.append("Résultats sur le test set :\n")
    lines.append("| Métrique | Valeur |\n")
    lines.append("|---|---:|\n")
    lines.append(f"| Precision | {best_model['precision_opt']:.4f} |\n")
    lines.append(f"| Recall | {best_model['recall_opt']:.4f} |\n")
    lines.append(f"| F1-score | {best_model['f1_opt']:.4f} |\n")
    lines.append(f"| ROC-AUC | {best_model['roc_auc']:.4f} |\n")
    lines.append(f"| Faux positifs | {int(best_model['FP_opt'])} |\n")
    lines.append(f"| Faux négatifs | {int(best_model['FN_opt'])} |\n")
    lines.append(f"| Vrais positifs | {int(best_model['TP_opt'])} |\n")
    lines.append(f"| Vrais négatifs | {int(best_model['TN_opt'])} |\n")
    lines.append(f"| Coût total optimal | {euros(best_model['cost_opt'])} |\n")
    lines.append(f"| Savings | {percent(best_model['savings'])} |\n")

    lines.append("---\n")

    lines.append("## 4. Lecture métier des résultats\n")
    lines.append(
        f"Le modèle final détecte {int(best_model['TP_opt'])} cas suspects sur les manipulations présentes "
        "dans le test set.\n"
    )
    lines.append(
        f"Il rate seulement {int(best_model['FN_opt'])} manipulation(s), ce qui est important dans un contexte "
        "réglementaire où le coût d'un faux négatif est très élevé.\n"
    )
    lines.append(
        f"Le modèle génère {int(best_model['FP_opt'])} fausse(s) alerte(s). Ce nombre reste acceptable, "
        "car le coût d'une enquête est faible par rapport au coût d'une manipulation non détectée.\n"
    )

    lines.append("La logique métier est donc :\n")
    lines.append("```text\n")
    lines.append("Il vaut mieux enquêter sur quelques faux positifs que rater une vraie manipulation.\n")
    lines.append("```\n")

    lines.append("---\n")

    lines.append("## 5. Comparaison des modèles après Threshold Moving\n")
    lines.append("| Modèle | Seuil optimal | Coût seuil 0.5 | Coût optimal | Savings | FN optimal | FP optimal |\n")
    lines.append("|---|---:|---:|---:|---:|---:|---:|\n")

    for _, row in threshold_df.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {row['best_threshold']:.6f} "
            f"| {euros(row['cost_05'])} "
            f"| {euros(row['cost_opt'])} "
            f"| {percent(row['savings'])} "
            f"| {int(row['FN_opt'])} "
            f"| {int(row['FP_opt'])} |\n"
        )

    lines.append("---\n")

    lines.append("## 6. Analyse de la calibration\n")
    lines.append(
        "La calibration est nécessaire car le Threshold Moving suppose que les probabilités prédites "
        "sont interprétables comme de vraies probabilités.\n"
    )

    lines.append("| Modèle | Méthode de calibration | Brier score | ROC-AUC | Log loss |\n")
    lines.append("|---|---|---:|---:|---:|\n")

    for _, row in calibration_df.iterrows():
        lines.append(
            f"| {row['model']} "
            f"| {row['calibration']} "
            f"| {row['brier_score']:.6f} "
            f"| {row['roc_auc']:.6f} "
            f"| {row['log_loss']:.6f} |\n"
        )

    lines.append("---\n")

    lines.append("## 7. Conclusion technique\n")
    lines.append(
        "Les résultats confirment que le déplacement du seuil permet de réduire fortement le coût total "
        "par rapport à un seuil classique de 0.5.\n"
    )

    lines.append("La démarche complète est :\n")
    lines.append("```text\n")
    lines.append("1. Entraîner un modèle probabiliste\n")
    lines.append("2. Calibrer les probabilités\n")
    lines.append("3. Chercher le seuil qui minimise le coût métier\n")
    lines.append("4. Évaluer uniquement sur le test set final\n")
    lines.append("```\n")

    lines.append(
        "Cette approche est plus adaptée qu'une simple maximisation de l'accuracy, car elle intègre "
        "directement l'asymétrie économique entre faux positifs et faux négatifs.\n"
    )

    lines.append("---\n")

    lines.append("## 8. Conclusion business\n")
    lines.append(
        "Le modèle final fournit un outil d'aide à la surveillance de marché. Il ne remplace pas "
        "l'analyste humain. Il priorise les alertes selon leur risque estimé et leur impact économique potentiel.\n"
    )
    lines.append(
        "L'intérêt principal du système est de transformer un modèle de classification classique "
        "en un outil de décision orienté coût, conforme à une logique de contrôle, de backtesting "
        "et de justification réglementaire.\n"
    )

    OUTPUT_FILE.write_text("".join(lines), encoding="utf-8")

    print(f"Rapport business généré : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()