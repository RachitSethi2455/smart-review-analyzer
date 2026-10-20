"""Metrics shared by all experiments."""
import json

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_recall_fscore_support

from src.config import FIGURES_DIR, RESULTS_DIR
from src.text_stats import confusion_matrix_figure


def compute_metrics(y_true, y_pred, average="macro"):
    p, r, f1, _ = precision_recall_fscore_support(y_true, y_pred, average=average, zero_division=0)
    return {"accuracy": round(accuracy_score(y_true, y_pred), 4),
            "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4)}


def report(name, y_true, y_pred, extra=None, labels=None, save_cm=True):
    """Print metrics, save confusion matrix PNG, return a result row."""
    metrics = compute_metrics(y_true, y_pred)
    print(f"\n=== {name} ===")
    print(classification_report(y_true, y_pred, zero_division=0, digits=4))
    if save_cm:
        fig = confusion_matrix_figure(y_true, y_pred, name, labels=labels)
        safe = "".join(c if c.isalnum() else "_" for c in name).strip("_").lower()
        fig.savefig(FIGURES_DIR / f"cm_{safe}.png", dpi=150)
    return {"model": name, **(extra or {}), **metrics}


def save_results(rows, filename):
    df = pd.DataFrame(rows)
    path = RESULTS_DIR / filename
    df.to_csv(path, index=False)
    print(f"\nSaved {path}")
    print(df.to_string(index=False))
    return df


def save_json(obj, filename):
    with open(RESULTS_DIR / filename, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
