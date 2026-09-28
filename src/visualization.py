import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import auc, roc_curve

# Set styling
sns.set_theme(style="whitegrid")
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 16,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "figure.titlesize": 18,
    }
)


def generate_and_save_plots(
    df_elo: pd.DataFrame, final_ratings: dict, model_results: dict, plots_dir: str
):
    """
    Generates all five analytical plots and saves them as PNGs in plots_dir.
    """
    os.makedirs(plots_dir, exist_ok=True)

    # 1. Elo Distribution Plot
    plt.figure(figsize=(10, 6))
    elos = list(final_ratings.values())
    sns.histplot(elos, bins=30, kde=True, color="#1f77b4", edgecolor="black", alpha=0.7)
    plt.axvline(
        1500, color="red", linestyle="--", linewidth=1.5, label="Rating Inicial (1500)"
    )
    plt.title("Distribuição de Ratings Elo dos Atletas de Judô")
    plt.xlabel("Rating Elo Final")
    plt.ylabel("Frequência (Atletas)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "elo_distribution.png"), dpi=300)
    plt.close()

    # Resolve names for top athletes
    # Build dictionary mapping athlete_id -> full name
    athlete_names = {}
    for row in df_elo.itertuples():
        athlete_names[row.athlete_blue_id] = (
            f"{row.blue_first_name} {row.blue_last_name}"
        )
        athlete_names[row.athlete_white_id] = (
            f"{row.white_first_name} {row.white_last_name}"
        )

    # 2. Top 10 Athletes by Elo
    top_athletes = sorted(final_ratings.items(), key=lambda x: x[1], reverse=True)[:10]
    top_names = [athlete_names.get(aid, f"Athlete {aid}") for aid, _ in top_athletes]
    top_elos = [elo for _, elo in top_athletes]

    plt.figure(figsize=(12, 7))
    sns.barplot(x=top_elos, y=top_names, hue=top_names, palette="viridis", legend=False)
    plt.xlim(min(top_elos) - 50, max(top_elos) + 50)
    plt.title("Top 10 Atletas de Judô com Maior Rating Elo")
    plt.xlabel("Rating Elo Final")
    plt.ylabel("Nome do Atleta")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "top_10_athletes.png"), dpi=300)
    plt.close()

    # 3. Elo Evolution over Time for the top athlete
    top_athlete_id = top_athletes[0][0]
    top_athlete_name = athlete_names.get(top_athlete_id, "Top Athlete")

    # Filter matches where the top athlete participated
    athlete_matches = df_elo[
        (df_elo["athlete_blue_id"] == top_athlete_id)
        | (df_elo["athlete_white_id"] == top_athlete_id)
    ].copy()

    # Extract ratings chronological timeline
    dates = []
    ratings_history = []

    for row in athlete_matches.itertuples():
        dates.append(row.date)
        if row.athlete_blue_id == top_athlete_id:
            ratings_history.append(row.blue_post_elo)
        else:
            ratings_history.append(row.white_post_elo)

    plt.figure(figsize=(12, 6))
    plt.plot(
        dates, ratings_history, marker="o", markersize=3, color="#2ca02c", linewidth=1.5
    )
    plt.title(f"Evolução do Rating Elo de {top_athlete_name}")
    plt.xlabel("Data")
    plt.ylabel("Rating Elo")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "elo_evolution_top_athlete.png"), dpi=300)
    plt.close()

    # 4. Elo Difference vs Win Rate (Calibration Plot)
    # We bin elo_diff = blue_pre_elo - white_pre_elo
    df_elo["elo_diff_bin"] = pd.cut(
        df_elo["blue_pre_elo"] - df_elo["white_pre_elo"], bins=np.arange(-600, 601, 100)
    )

    df_win_rate = (
        df_elo.groupby("elo_diff_bin", observed=False)
        .apply(
            lambda g: (g["winner_color"] == "blue").mean() if len(g) > 20 else np.nan
        )
        .reset_index()
    )
    df_win_rate.columns = ["elo_diff_bin", "win_rate"]
    df_win_rate = df_win_rate.dropna()

    # Format labels for x-axis
    bin_labels = [
        f"[{int(b.left)}, {int(b.right)})" for b in df_win_rate["elo_diff_bin"]
    ]

    plt.figure(figsize=(12, 6))
    sns.barplot(
        x=bin_labels,
        y=df_win_rate["win_rate"],
        hue=bin_labels,
        palette="coolwarm",
        legend=False,
    )
    plt.axhline(0.5, color="gray", linestyle="--", linewidth=1)
    plt.title("Diferença de Elo (Blue - White) vs. Taxa de Vitória do Blue")
    plt.xlabel("Diferença de Rating Elo Pré-Luta")
    plt.ylabel("Taxa de Vitória do Atleta Blue")
    plt.ylim(0, 1.05)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "elo_diff_vs_winrate.png"), dpi=300)
    plt.close()

    # 5. ROC Curve Plot (Model vs. Baseline)
    y_test = model_results["y_test"]
    y_prob = model_results["y_prob"]
    elo_diff_test = model_results["elo_diff_test"]

    # Compute ROC curves
    fpr_model, tpr_model, _ = roc_curve(y_test, y_prob)
    roc_auc_model = auc(fpr_model, tpr_model)

    # For baseline, we scale elo_diff_test to act as probability estimation (via sigmoid expected score)
    expected_scores_test = 1.0 / (1.0 + 10.0 ** (-elo_diff_test / 400.0))
    fpr_base, tpr_base, _ = roc_curve(y_test, expected_scores_test)
    roc_auc_base = auc(fpr_base, tpr_base)

    plt.figure(figsize=(10, 8))
    plt.plot(
        fpr_model,
        tpr_model,
        color="darkorange",
        lw=2,
        label=f"Random Forest Classifier (AUC = {roc_auc_model:.4f})",
    )
    plt.plot(
        fpr_base,
        tpr_base,
        color="navy",
        lw=2,
        linestyle="--",
        label=f"Baseline Elo Diferença (AUC = {roc_auc_base:.4f})",
    )
    plt.plot([0, 1], [0, 1], color="gray", lw=1.5, linestyle=":")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("Taxa de Falsos Positivos (FPR)")
    plt.ylabel("Taxa de Verdadeiros Positivos (TPR)")
    plt.title("Curva ROC - Modelo de Machine Learning vs. Baseline Elo")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_roc_curve.png"), dpi=300)
    plt.close()

    print(f"All plots successfully generated and saved to: {plots_dir}")
