import pandas as pd
from pathlib import Path


RESULTS_DIR = Path("reports/results")

FINAL_COMPARISON_FILE = RESULTS_DIR / "final_extended_model_comparison.csv"
OUTPUT_FILE = RESULTS_DIR / "final_results_analysis.md"

C_FN = 15_000_000
C_FP = 7_000


def euros(x):
    return f"{x:,.0f} €".replace(",", " ")


def main():
    df = pd.read_csv(FINAL_COMPARISON_FILE)
    df = df.sort_values("cost").reset_index(drop=True)

    best = df.iloc[0]

    threshold_models = df[df["method"] == "Calibration + Threshold Moving"]
    best_tm = threshold_models.sort_values("cost").iloc[0]

    smote_models = df[df["method"] == "SMOTE + Threshold 0.5"]
    best_smote = smote_models.sort_values("cost").iloc[0]

    lines = []

    lines.append("# Analyse finale des résultats\n\n")

    lines.append("## 1. Rappel de l'objectif\n\n")
    lines.append(
        "L'objectif du projet est de détecter les manipulations de marché en minimisant le coût total des erreurs. "
        "Dans ce contexte, toutes les erreurs n'ont pas la même gravité.\n\n"
    )

    lines.append("La fonction de coût utilisée est :\n\n")
    lines.append("```text\n")
    lines.append("Coût total = c_FN × FN + c_FP × FP\n")
    lines.append("```\n\n")

    lines.append("avec :\n\n")
    lines.append("```text\n")
    lines.append(f"c_FN = {euros(C_FN)}\n")
    lines.append(f"c_FP = {euros(C_FP)}\n")
    lines.append("```\n\n")

    lines.append(
        "Un faux négatif correspond à une manipulation réelle non détectée. "
        "C'est l'erreur la plus coûteuse. Un faux positif correspond à une fausse alerte, "
        "dont le coût est limité au traitement de l'enquête.\n\n"
    )

    lines.append("---\n\n")

    lines.append("## 2. Résultat global\n\n")
    lines.append("Le meilleur modèle en coût total sur le test set est :\n\n")
    lines.append("```text\n")
    lines.append(f"{best['model']} — {best['method']}\n")
    lines.append("```\n\n")

    lines.append("Ses résultats sont :\n\n")
    lines.append("| Métrique | Valeur |\n")
    lines.append("|---|---:|\n")
    lines.append(f"| Accuracy | {best['accuracy']:.4f} |\n")
    lines.append(f"| Precision | {best['precision']:.4f} |\n")
    lines.append(f"| Recall | {best['recall']:.4f} |\n")
    lines.append(f"| F1-score | {best['f1_score']:.4f} |\n")
    lines.append(f"| ROC-AUC | {best['roc_auc']:.4f} |\n")
    lines.append(f"| Faux positifs | {int(best['FP'])} |\n")
    lines.append(f"| Faux négatifs | {int(best['FN'])} |\n")
    lines.append(f"| Vrais positifs | {int(best['TP'])} |\n")
    lines.append(f"| Vrais négatifs | {int(best['TN'])} |\n")
    lines.append(f"| Coût total | {euros(best['cost'])} |\n\n")

    lines.append(
        "Ce résultat signifie que le modèle détecte toutes les manipulations du test set, "
        "au prix de quelques fausses alertes. Comme le coût d'un faux positif est faible par rapport "
        "au coût d'un faux négatif, cette stratégie est économiquement très favorable.\n\n"
    )

    lines.append("---\n\n")

    lines.append("## 3. Comparaison entre SMOTE et Threshold Moving\n\n")

    lines.append("### 3.1 Meilleur modèle SMOTE\n\n")
    lines.append("| Élément | Valeur |\n")
    lines.append("|---|---:|\n")
    lines.append(f"| Modèle | {best_smote['model']} |\n")
    lines.append(f"| Méthode | {best_smote['method']} |\n")
    lines.append(f"| Coût | {euros(best_smote['cost'])} |\n")
    lines.append(f"| FN | {int(best_smote['FN'])} |\n")
    lines.append(f"| FP | {int(best_smote['FP'])} |\n")
    lines.append(f"| Recall | {best_smote['recall']:.4f} |\n")
    lines.append(f"| Precision | {best_smote['precision']:.4f} |\n\n")

    lines.append("### 3.2 Meilleur modèle Threshold Moving\n\n")
    lines.append("| Élément | Valeur |\n")
    lines.append("|---|---:|\n")
    lines.append(f"| Modèle | {best_tm['model']} |\n")
    lines.append(f"| Méthode | {best_tm['method']} |\n")
    lines.append(f"| Seuil optimal | {best_tm['threshold']:.6f} |\n")
    lines.append(f"| Coût | {euros(best_tm['cost'])} |\n")
    lines.append(f"| FN | {int(best_tm['FN'])} |\n")
    lines.append(f"| FP | {int(best_tm['FP'])} |\n")
    lines.append(f"| Recall | {best_tm['recall']:.4f} |\n")
    lines.append(f"| Precision | {best_tm['precision']:.4f} |\n\n")

    lines.append(
        "Dans cette expérimentation, SMOTE obtient le coût le plus faible, principalement parce qu'il ne rate aucune manipulation. "
        "Le meilleur modèle Threshold Moving rate une manipulation, ce qui entraîne immédiatement un coût très élevé à cause de la valeur de c_FN.\n\n"
    )

    lines.append("---\n\n")

    lines.append("## 4. Interprétation critique\n\n")
    lines.append(
        "Le résultat ne signifie pas que SMOTE est toujours supérieur. Il signifie que, sur ce test set précis, "
        "la stratégie SMOTE + Random Forest a permis d'obtenir zéro faux négatif.\n\n"
    )

    lines.append("Cependant, SMOTE possède des limites importantes :\n\n")
    lines.append("- il modifie artificiellement la distribution d'apprentissage ;\n")
    lines.append("- il crée des observations synthétiques qui peuvent être peu réalistes économiquement ;\n")
    lines.append("- il ne tient pas directement compte de la matrice de coûts ;\n")
    lines.append("- il optimise un problème statistique, pas directement un problème de décision métier.\n\n")

    lines.append(
        "Threshold Moving reste plus propre du point de vue décisionnel, car le seuil est explicitement relié "
        "aux coûts économiques des erreurs. Cette approche est donc plus facile à justifier dans un cadre réglementaire "
        "ou devant un analyste métier.\n\n"
    )

    lines.append("---\n\n")

    lines.append("## 5. Tableau final de comparaison\n\n")
    lines.append("| Rang | Modèle | Méthode | Seuil | Precision | Recall | F1-score | Coût | FP | FN | TP | TN |\n")
    lines.append("|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")

    for i, row in df.iterrows():
        lines.append(
            f"| {i + 1} "
            f"| {row['model']} "
            f"| {row['method']} "
            f"| {row['threshold']:.6f} "
            f"| {row['precision']:.4f} "
            f"| {row['recall']:.4f} "
            f"| {row['f1_score']:.4f} "
            f"| {euros(row['cost'])} "
            f"| {int(row['FP'])} "
            f"| {int(row['FN'])} "
            f"| {int(row['TP'])} "
            f"| {int(row['TN'])} |\n"
        )

    lines.append("\n---\n\n")

    lines.append("## 6. Conclusion finale\n\n")
    lines.append(
        "La meilleure performance empirique est obtenue avec Random Forest + SMOTE, car cette méthode atteint un rappel parfait "
        "sur le test set et évite totalement les faux négatifs.\n\n"
    )

    lines.append(
        "Cependant, la méthode centrale du projet reste Calibration + Threshold Moving, car elle formalise directement "
        "la décision à partir de la théorie de Bayes et de la matrice de coûts. Elle permet de transformer un modèle probabiliste "
        "en outil de décision économique.\n\n"
    )

    lines.append("La conclusion correcte à présenter est donc :\n\n")
    lines.append("```text\n")
    lines.append("SMOTE domine empiriquement sur ce test set.\n")
    lines.append("Threshold Moving reste plus justifiable théoriquement, économiquement et réglementairement.\n")
    lines.append("La comparaison montre l'écart entre performance observée et robustesse décisionnelle.\n")
    lines.append("```\n")

    OUTPUT_FILE.write_text("".join(lines), encoding="utf-8")

    print(f"Analyse finale générée : {OUTPUT_FILE}")


if __name__ == "__main__":
    main()