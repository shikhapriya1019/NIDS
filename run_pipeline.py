#!/usr/bin/env python3
"""
run_pipeline.py
===============
End-to-end supervised machine learning pipeline for cross-dataset
generalisation experiments on Network Intrusion Detection Systems (NIDS).

Models trained on CICIDS2017, evaluated on UNSW-NB15 and CSE-CIC-IDS2018.

Usage
-----
    # Quick test with auto-generated mock data:
    python run_pipeline.py --generate-data

    # With real CSV files in data/raw/:
    python run_pipeline.py --data-dir data/raw

    # Custom settings:
    python run_pipeline.py --data-dir data/raw --output-dir results --cv-folds 5
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised NIDS Pipeline — Cross-Dataset Generalisation"
    )
    parser.add_argument("--data-dir",    type=str, default="data/raw",
                        help="Directory with raw CSV datasets.")
    parser.add_argument("--output-dir",  type=str, default="output",
                        help="Directory for metrics, models, and plots.")
    parser.add_argument("--cv-folds",    type=int, default=5,
                        help="Number of Stratified K-Fold CV splits (default 5).")
    parser.add_argument("--n-iter",      type=int, default=10,
                        help="RandomizedSearchCV iterations (default 10).")
    parser.add_argument("--k-features",  type=int, default=20,
                        help="Number of features to select (default 20).")
    parser.add_argument("--generate-data", action="store_true",
                        help="Generate mock CSV files before running.")
    parser.add_argument("--skip-knn",   action="store_true",
                        help="Skip KNN (slow on large datasets).")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------------
def extract_labels(df: pd.DataFrame, dataset_type: str) -> np.ndarray:
    """
    Convert raw label column to binary integers (0 = benign, 1 = attack).

    Dataset-specific rules
    ----------------------
    CICIDS2017        : 'BENIGN' → 0, all others → 1
    UNSW-NB15         : already 0 / 1
    CSE-CIC-IDS2018   : 'Benign' → 0, all others → 1
    """
    if "label" not in df.columns:
        raise ValueError(
            f"Label column missing. Available: {list(df.columns)}"
        )

    raw = df["label"].copy()

    if dataset_type == "unswnb15":
        return raw.astype(int).values

    # String-label datasets
    normalised = raw.astype(str).str.strip().str.lower()
    return np.where(normalised == "benign", 0, 1)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def main() -> None:
    args    = parse_args()
    t_start = time.time()

    data_dir   = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    plots_dir  = output_dir / "plots"
    models_dir = output_dir / "saved_models"
    plots_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 65)
    logger.info("Supervised NIDS Pipeline — START")
    logger.info("=" * 65)
    logger.info("Data dir    : %s", data_dir)
    logger.info("Output dir  : %s", output_dir)
    logger.info("CV folds    : %d", args.cv_folds)
    logger.info("Search iters: %d", args.n_iter)
    logger.info("K features  : %d", args.k_features)

    # -----------------------------------------------------------------------
    # Step 1 — Optional mock data generation
    # -----------------------------------------------------------------------
    if args.generate_data:
        logger.info("[Step 1] Generating mock datasets …")
        from data.generate_mock_data import generate_all
        generate_all(str(data_dir.parent))
        logger.info("[Step 1] Mock data written to %s", data_dir)
    else:
        logger.info("[Step 1] Skipped — using existing CSV files.")

    # -----------------------------------------------------------------------
    # Step 2 — Load datasets
    # -----------------------------------------------------------------------
    logger.info("[Step 2] Loading datasets …")

    dataset_files = {
        "cicids2017_train":    ("cicids2017_train.csv",    "cicids2017"),
        "cicids2017_test":     ("cicids2017_test.csv",     "cicids2017"),
        "unswnb15_test":       ("unswnb15_test.csv",       "unswnb15"),
        "cse_cic_ids2018_test":("cse_cic_ids2018_test.csv","cse_cic_ids2018"),
    }

    raw_frames:    dict[str, pd.DataFrame] = {}
    dataset_types: dict[str, str]          = {}

    for key, (fname, ds_type) in dataset_files.items():
        path = data_dir / fname
        if not path.exists():
            logger.error("File not found: %s", path)
            sys.exit(1)
        raw_frames[key]    = pd.read_csv(path)
        dataset_types[key] = ds_type
        logger.info("  Loaded %-34s  %d rows × %d cols",
                    fname, *raw_frames[key].shape)

    # -----------------------------------------------------------------------
    # Step 3 — Clean all datasets
    # -----------------------------------------------------------------------
    logger.info("[Step 3] Cleaning datasets …")
    from preprocessing.clean import clean_dataframe

    cleaned: dict[str, pd.DataFrame] = {}
    for key, df in raw_frames.items():
        cleaned[key] = clean_dataframe(df)
        logger.info("  %-34s  %d rows × %d cols after cleaning",
                    key, *cleaned[key].shape)

    # -----------------------------------------------------------------------
    # Step 4 — Extract labels & align features
    # -----------------------------------------------------------------------
    logger.info("[Step 4] Extracting labels and aligning features …")
    from preprocessing.align_features import align_features

    labels:  dict[str, np.ndarray]  = {}
    aligned: dict[str, pd.DataFrame] = {}

    for key, df in cleaned.items():
        labels[key] = extract_labels(df, dataset_types[key])
        logger.info("  %-34s  benign=%d  attack=%d",
                    key,
                    int((labels[key] == 0).sum()),
                    int((labels[key] == 1).sum()))
        feature_df    = df.drop(columns=["label"], errors="ignore")
        aligned[key]  = align_features(feature_df, dataset_type=dataset_types[key])
        logger.info("  %-34s  %d aligned features", key, aligned[key].shape[1])

    # -----------------------------------------------------------------------
    # Step 5 — Scale (fit on FULL training set — benign + attack)
    # -----------------------------------------------------------------------
    logger.info("[Step 5] Fitting MinMaxScaler on full CICIDS2017 training data …")
    from preprocessing.scale import fit_scaler, transform

    scaler = fit_scaler(aligned["cicids2017_train"])
    joblib.dump(scaler, models_dir / "scaler.pkl")
    logger.info("  Scaler fitted on %d training samples.", len(aligned["cicids2017_train"]))

    scaled: dict[str, np.ndarray] = {}
    for key, df in aligned.items():
        scaled[key] = transform(df, scaler)
        logger.info("  %-34s  scaled shape %s", key, scaled[key].shape)

    X_train = scaled["cicids2017_train"]
    y_train = labels["cicids2017_train"]

    # -----------------------------------------------------------------------
    # Step 6 — Feature selection (SelectKBest, fitted on training only)
    # -----------------------------------------------------------------------
    logger.info("[Step 6] Selecting top %d features …", args.k_features)
    from preprocessing.feature_selection import select_features

    test_arrays = {
        "cicids2017":      scaled["cicids2017_test"],
        "unswnb15":        scaled["unswnb15_test"],
        "cse_cic_ids2018": scaled["cse_cic_ids2018_test"],
    }

    X_train_sel, X_test_sel, selected_idx = select_features(
        X_train, y_train, test_arrays, k=args.k_features
    )
    logger.info("  Training shape after selection: %s", X_train_sel.shape)

    # -----------------------------------------------------------------------
    # Step 7 — Train supervised models with CV + hyperparameter tuning
    # -----------------------------------------------------------------------
    logger.info("[Step 7] Training supervised models …")
    from models import (
        RandomForestModel,
        XGBoostModel,
        DecisionTreeModel,
        LogisticRegressionModel,
        KNNModel,
        MLPModel,
        _XGBOOST_AVAILABLE,
    )

    model_list = [
        RandomForestModel(n_iter=args.n_iter, cv_folds=args.cv_folds),
        DecisionTreeModel(cv_folds=args.cv_folds),
        LogisticRegressionModel(cv_folds=args.cv_folds),
        MLPModel(n_iter=args.n_iter, cv_folds=args.cv_folds),
    ]
    if _XGBOOST_AVAILABLE:
        model_list.insert(1, XGBoostModel(n_iter=args.n_iter, cv_folds=args.cv_folds))
    else:
        logger.warning("XGBoost not installed — skipping. Install with: pip install xgboost")
    if not args.skip_knn:
        model_list.insert(-1, KNNModel(cv_folds=args.cv_folds))

    trained_models = []
    for model in model_list:
        t0 = time.time()
        logger.info("  Training %s …", model.name)
        model.fit(X_train_sel, y_train)
        elapsed = time.time() - t0
        logger.info("  %s trained in %.1f s — best params: %s",
                    model.name, elapsed, model.best_params_)
        joblib.dump(model, models_dir / f"{model.name.replace(' ', '_')}.pkl")
        trained_models.append(model)

    # -----------------------------------------------------------------------
    # Step 8 — Evaluate on all test sets
    # -----------------------------------------------------------------------
    logger.info("[Step 8] Evaluating models on all test datasets …")
    from evaluation.metrics import compute_metrics, format_metrics_table

    test_sets = {
        "cicids2017":      (X_test_sel["cicids2017"],      labels["cicids2017_test"]),
        "unswnb15":        (X_test_sel["unswnb15"],        labels["unswnb15_test"]),
        "cse_cic_ids2018": (X_test_sel["cse_cic_ids2018"], labels["cse_cic_ids2018_test"]),
    }

    all_metrics: dict[str, dict] = {}
    raw_results: dict[str, dict] = {}

    for ds_name, (X_test, y_test) in test_sets.items():
        all_metrics[ds_name] = {}
        raw_results[ds_name] = {}

        for model in trained_models:
            y_pred   = model.predict(X_test)
            y_scores = model.predict_proba(X_test)
            metrics  = compute_metrics(y_test, y_pred, y_scores)
            all_metrics[ds_name][model.name] = metrics
            raw_results[ds_name][model.name] = {
                "y_true":   y_test,
                "y_scores": y_scores,
            }
            logger.info(
                "    %-22s on %-20s  F1=%.4f  ROC-AUC=%.4f",
                model.name, ds_name,
                metrics["f1_score"], metrics["roc_auc"],
            )

    # Print formatted summary tables
    for ds_name, model_metrics in all_metrics.items():
        print(f"\n{'='*65}")
        print(f"  Results — {ds_name.upper()}")
        print(f"{'='*65}")
        print(format_metrics_table(model_metrics))

    # -----------------------------------------------------------------------
    # Step 9 — Visualisations
    # -----------------------------------------------------------------------
    logger.info("[Step 9] Generating plots …")
    from evaluation.visualise import (
        plot_roc_curves,
        plot_pr_curves,
        plot_f1_comparison,
        plot_pca_datasets,
        plot_score_distributions,
    )

    for ds_name, model_results in raw_results.items():
        plot_roc_curves(model_results,
                        str(plots_dir / f"roc_{ds_name}.png"))
        plot_pr_curves(model_results,
                       str(plots_dir / f"pr_{ds_name}.png"))
        plot_score_distributions(model_results,
                                 str(plots_dir / f"scores_{ds_name}.png"))
        logger.info("  Saved ROC / PR / score plots for %s", ds_name)

    plot_f1_comparison(all_metrics,
                       str(plots_dir / "f1_comparison.png"))
    logger.info("  Saved F1 comparison bar chart.")

    pca_datasets = {
        ds: pd.DataFrame(X_test_sel[ds]) for ds in test_sets
    }
    plot_pca_datasets(pca_datasets,
                      str(plots_dir / "pca_datasets.png"))
    logger.info("  Saved PCA dataset overlay.")

    # -----------------------------------------------------------------------
    # Step 10 — Save metrics JSON
    # -----------------------------------------------------------------------
    logger.info("[Step 10] Saving metrics to JSON …")

    def _serialise(obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return obj

    serialisable = {
        ds: {
            m: {k: _serialise(v) for k, v in met.items() if k != "confusion_matrix"}
            for m, met in models_dict.items()
        }
        for ds, models_dict in all_metrics.items()
    }

    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serialisable, f, indent=2, default=_serialise)
    logger.info("  Metrics saved to %s", metrics_path)

    # -----------------------------------------------------------------------
    # Done
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    logger.info("=" * 65)
    logger.info("Pipeline completed in %.1f s", elapsed)
    logger.info("Results written to: %s", output_dir.resolve())
    logger.info("=" * 65)


if __name__ == "__main__":
    main()
