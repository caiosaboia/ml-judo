import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def train_lightgbm_model(X: pd.DataFrame, y: pd.Series, split_ratio: float = 0.8) -> dict:
    """
    Trains a LightGBM Classifier chronologically and evaluates performance against baseline.
    """
    n_samples = len(X)
    split_idx = int(n_samples * split_ratio)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = LGBMClassifier(
        n_estimators=150,
        learning_rate=0.05,
        max_depth=6,
        num_leaves=31,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = np.asarray(model.predict_proba(X_test))[:, 1]  # pyright: ignore[reportCallIssue, reportArgumentType]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    conf_mat = confusion_matrix(y_test, y_pred)

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    feature_ranking = []

    for f in range(min(10, X.shape[1])):
        feat_name = str(X.columns[indices[f]])
        feat_imp = float(importances[indices[f]])
        feature_ranking.append((feat_name, feat_imp))

    return {
        "model": model,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": roc_auc,
            "confusion_matrix": conf_mat.tolist()
        },
        "feature_ranking": feature_ranking,
        "y_test": y_test,
        "y_prob": y_prob
    }

def parse_victory_method_target(df: pd.DataFrame) -> pd.Series:
    """
    Categorizes victory method into 'ippon', 'wazari', or 'shido' (decision/penalties).
    """
    methods = []
    for row in df.itertuples():
        b_ippon = getattr(row, 'blue_ippon', 0)
        w_ippon = getattr(row, 'white_ippon', 0)
        b_wazari = getattr(row, 'blue_wazari', 0)
        w_wazari = getattr(row, 'white_wazari', 0)

        try:
            b_ippon = int(b_ippon) if pd.notna(b_ippon) else 0
            w_ippon = int(w_ippon) if pd.notna(w_ippon) else 0
            b_wazari = int(b_wazari) if pd.notna(b_wazari) else 0
            w_wazari = int(w_wazari) if pd.notna(w_wazari) else 0
        except (ValueError, TypeError):
            b_ippon, w_ippon, b_wazari, w_wazari = 0, 0, 0, 0

        if b_ippon > 0 or w_ippon > 0:
            methods.append('ippon')
        elif b_wazari > 0 or w_wazari > 0:
            methods.append('wazari')
        else:
            methods.append('shido')

    return pd.Series(methods, index=df.index)

def train_victory_method_model(X: pd.DataFrame, y_method: pd.Series, split_ratio: float = 0.8) -> dict:
    """
    Trains a Multi-Class LightGBM Classifier to predict victory method (ippon, wazari, shido).
    """
    n_samples = len(X)
    split_idx = int(n_samples * split_ratio)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y_method.iloc[:split_idx], y_method.iloc[split_idx:]

    model = LGBMClassifier(
        n_estimators=100,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        verbosity=-1,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

    return {
        "model": model,
        "metrics": {
            "accuracy": acc,
            "report": report
        },
        "classes": list(model.classes_),
        "y_test": y_test,
        "y_prob": y_prob
    }
