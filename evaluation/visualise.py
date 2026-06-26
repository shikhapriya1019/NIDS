"""
evaluation/visualise.py
Plotting functions: ROC curves, PR curves, F1 bar chart,
PCA dataset overlay, and score distribution histograms.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from sklearn.decomposition import PCA


sns.set_theme(style="whitegrid", palette="tab10")
FIGSIZE = (9, 6)


# ---------------------------------------------------------------------------
# ROC Curves
# ---------------------------------------------------------------------------
def plot_roc_curves(model_results: dict, save_path: str) -> None:
    """
    Plot ROC curves for all models on a single dataset.

    Parameters
    ----------
    model_results : dict
        {model_name: {'y_true': array, 'y_scores': array}}
    save_path : str
        File path for the saved figure.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for model_name, res in model_results.items():
        fpr, tpr, _ = roc_curve(res["y_true"], res["y_scores"])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{model_name}  (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves", fontsize=14)
    ax.legend(loc="lower right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# PR Curves
# ---------------------------------------------------------------------------
def plot_pr_curves(model_results: dict, save_path: str) -> None:
    """
    Plot Precision-Recall curves for all models on a single dataset.
    """
    fig, ax = plt.subplots(figsize=FIGSIZE)
    for model_name, res in model_results.items():
        prec, rec, _ = precision_recall_curve(res["y_true"], res["y_scores"])
        pr_auc = auc(rec, prec)
        ax.plot(rec, prec, label=f"{model_name}  (AUC = {pr_auc:.3f})")
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves", fontsize=14)
    ax.legend(loc="upper right", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# F1 Comparison Bar Chart
# ---------------------------------------------------------------------------
def plot_f1_comparison(all_metrics: dict, save_path: str) -> None:
    """
    Grouped bar chart of F1-scores across datasets and models.

    Parameters
    ----------
    all_metrics : dict
        {dataset_name: {model_name: metrics_dict}}
    save_path : str
        Output file path.
    """
    datasets = list(all_metrics.keys())
    models   = list(next(iter(all_metrics.values())).keys())
    x        = np.arange(len(models))
    width    = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    for i, ds in enumerate(datasets):
        f1_vals = [all_metrics[ds][m]["f1_score"] for m in models]
        ax.bar(x + i * width, f1_vals, width, label=ds)

    ax.set_xticks(x + width * (len(datasets) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right", fontsize=10)
    ax.set_ylabel("F1-Score", fontsize=12)
    ax.set_title("F1-Score Comparison Across Datasets", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# PCA Dataset Overlay
# ---------------------------------------------------------------------------
def plot_pca_datasets(dataset_arrays: dict, save_path: str) -> None:
    """
    2-D PCA scatter plot overlaying all three dataset distributions.

    Parameters
    ----------
    dataset_arrays : dict
        {dataset_name: np.ndarray of shape (n_samples, n_features)}
    save_path : str
        Output file path.
    """
    pca = PCA(n_components=2, random_state=42)
    all_data = np.vstack(list(dataset_arrays.values()))
    # Subsample for speed if large
    if len(all_data) > 20_000:
        idx = np.random.default_rng(42).choice(len(all_data), 20_000, replace=False)
        all_data = all_data[idx]

    pca.fit(all_data)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    offset = 0
    for ds_name, arr in dataset_arrays.items():
        if len(arr) > 5_000:
            sample_idx = np.random.default_rng(0).choice(len(arr), 5_000, replace=False)
            arr = arr[sample_idx]
        transformed = pca.transform(arr)
        ax.scatter(transformed[:, 0], transformed[:, 1],
                   alpha=0.3, s=4, label=ds_name)

    ax.set_xlabel("PC 1", fontsize=12)
    ax.set_ylabel("PC 2", fontsize=12)
    ax.set_title("PCA Feature Space — Dataset Overlap", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Score Distribution
# ---------------------------------------------------------------------------
def plot_score_distributions(model_results: dict, save_path: str) -> None:
    """
    Histogram of predicted probabilities for benign vs attack,
    for each model.
    """
    n_models = len(model_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 4), sharey=True)
    if n_models == 1:
        axes = [axes]

    for ax, (model_name, res) in zip(axes, model_results.items()):
        y_true   = res["y_true"]
        y_scores = res["y_scores"]
        ax.hist(y_scores[y_true == 0], bins=40, alpha=0.6,
                label="Benign", color="steelblue", density=True)
        ax.hist(y_scores[y_true == 1], bins=40, alpha=0.6,
                label="Attack", color="tomato", density=True)
        ax.set_title(model_name, fontsize=10)
        ax.set_xlabel("P(attack)", fontsize=9)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Density", fontsize=10)
    fig.suptitle("Score Distributions — Benign vs Attack", fontsize=13)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
