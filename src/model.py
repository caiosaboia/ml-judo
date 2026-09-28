import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_and_evaluate_model(
    X: pd.DataFrame, y: pd.Series, split_ratio: float = 0.8
) -> dict:
    """
    Splits the data chronologically, trains a Scikit-Learn RandomForestClassifier,
    evaluates its performance, and compares it to the Elo-only baseline.
    """
    n_samples = len(X)
    split_idx = int(n_samples * split_ratio)

    # Chronological Split
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    print(f"Train size: {len(X_train)} matches, Test size: {len(X_test)} matches.")

    # Baseline: Higher pre-match Elo wins
    # (If elo_diff > 0, Blue is predicted to win. If elo_diff == 0, random or 1)
    # Let's count how often the player with higher Elo wins in the test set.
    elo_diff_test = X_test["elo_diff"]
    y_pred_baseline = (elo_diff_test > 0).astype(int)
    # For cases where elo_diff is exactly 0, baseline predicts 1 (blue wins) or we can match actual
    # to keep it simple: predicting Blue wins if elo_diff >= 0
    y_pred_baseline = (elo_diff_test >= 0).astype(int)

    baseline_acc = accuracy_score(y_test, y_pred_baseline)
    baseline_roc_auc = roc_auc_score(y_test, elo_diff_test)

    print("\n--- Baseline (Higher Elo Wins) ---")
    print(f"Accuracy:  {baseline_acc:.4f}")
    print(f"ROC AUC:   {baseline_roc_auc:.4f}")

    # Model: RandomForestClassifier
    # Limit max_depth to prevent overfitting and make it faster
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=42, n_jobs=-1
    )
    model.fit(X_train, y_train)

    # Predictions
    y_pred = model.predict(X_test)
    y_prob = np.asarray(model.predict_proba(X_test))[:, 1]  # pyright: ignore[reportCallIssue, reportArgumentType]

    # Calculate Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_mat = confusion_matrix(y_test, y_pred)

    print("\n--- Machine Learning Model (Random Forest) ---")
    print(f"Accuracy:  {acc:.4f} (Change vs Baseline: {acc - baseline_acc:+.4f})")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"ROC AUC:   {roc_auc:.4f}")

    # Feature Importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_ranking = []

    print("\nFeature Importances:")
    for f in range(min(10, X.shape[1])):
        feat_name = X.columns[indices[f]]
        feat_imp = importances[indices[f]]
        feature_ranking.append((feat_name, feat_imp))
        print(f"  {f + 1}. {feat_name:25s}: {feat_imp:.4f}")

    return {
        "model": model,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_mat.tolist(),
        },
        "baseline": {"accuracy": baseline_acc, "roc_auc": baseline_roc_auc},
        "feature_ranking": feature_ranking,
        "y_test": y_test,
        "y_prob": y_prob,
        "elo_diff_test": elo_diff_test,
    }
