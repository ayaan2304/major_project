"""
Figures: class imbalance, metric bars, SHAP-related distributions.

What / why
----------
Visual evidence for why accuracy is the wrong headline metric and for
EP-IAIL accept/reject distributions.

Leakage risks
-------------
Plot functions must not retrain models. They only consume arrays already
computed on the intended split.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_theme(style="whitegrid", context="talk")


def plot_class_imbalance(n_legit: int, n_fraud: int, path: Path, title: str = "Class counts") -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    labels = ["Legitimate", "Fraud"]
    counts = [n_legit, n_fraud]
    axes[0].bar(labels, counts, color=["#4C78A8", "#F58518"])
    axes[0].set_ylabel("Count")
    axes[0].set_title(title)
    axes[0].set_yscale("log")
    axes[1].pie(counts, labels=labels, autopct="%.3f%%", colors=["#4C78A8", "#F58518"], startangle=90)
    axes[1].set_title("Class share")
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_metric_bars(df: pd.DataFrame, metric_cols: list[str], path: Path, title: str) -> None:
    long = df.melt(id_vars=["model"], value_vars=metric_cols, var_name="metric", value_name="value")
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(data=long, x="metric", y="value", hue="model", ax=ax)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_score_hist(values: np.ndarray, path: Path, title: str, xlabel: str, vline: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(values, bins=40, color="#4C78A8", edgecolor="white")
    if vline is not None:
        ax.axvline(vline, color="#E45756", linestyle="--", label=f"threshold={vline:.3f}")
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_reliability(y_true, y_prob, path: Path, n_bins: int = 12, title: str = "Reliability") -> None:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    centers, accs, confs = [], [], []
    for i in range(n_bins):
        m = (y_prob >= bins[i]) & ((y_prob < bins[i + 1]) if i < n_bins - 1 else (y_prob <= bins[i + 1]))
        if m.sum() == 0:
            continue
        centers.append(0.5 * (bins[i] + bins[i + 1]))
        accs.append(y_true[m].mean())
        confs.append(y_prob[m].mean())
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="grey")
    ax.plot(confs, accs, "o-", color="#4C78A8")
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed fraud frequency")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
