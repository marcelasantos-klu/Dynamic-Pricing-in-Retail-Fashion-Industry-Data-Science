"""
Train Airbnb revenue models and run additional experiments.

Pipeline overview:
- Runs both with and without outlier filtering (IQR) for each split strategy.
- Regression: multiple model families with log-target training, global + per-city metrics, and diagnostic plots.
- Classification: high-revenue detection (median threshold computed on the train set) with standard metrics.
- Clustering: KMeans (k in {2,3,4}) with inertia/silhouette and summaries for interpretation.
- Feature engineering is shared and leakage-aware (fit on train, apply to test).
- Outputs (plots, models, logs) live under plots&models/ in split-specific subfolders.
"""
from __future__ import annotations

import os
import sys
import ctypes
from pathlib import Path
from typing import List

import pandas as pd

from config import LOG_PATH, OUTLIER_BASE, SPLIT_STRATEGIES
from data_io import load_data, Tee
from models_regression import train_for_split
from models_classification import run_classification_experiment
from models_clustering import run_clustering_experiment


def load_libomp_if_available() -> None:
    """Try to load libomp on macOS for gradient boosting libraries."""
    candidates = [
        Path("/opt/homebrew/opt/libomp/lib/libomp.dylib"),
        Path("/usr/local/opt/libomp/lib/libomp.dylib"),
        Path.home() / "Library" / "Python" / "3.9" / "lib" / "python" / "site-packages" / "torch" / "lib" / "libomp.dylib",
    ]
    for path in candidates:
        if path.exists():
            try:
                ctypes.cdll.LoadLibrary(str(path))
                os.environ.setdefault("DYLD_LIBRARY_PATH", str(path.parent))
                os.environ.setdefault("DYLD_FALLBACK_LIBRARY_PATH", str(path.parent))
                print(f"Loaded libomp from: {path}")
                return
            except OSError:
                pass
    print("Warning: libomp not found. LightGBM may fail on macOS.")


def main() -> None:
    """Orchestrate regression, classification, and clustering experiments."""
    load_libomp_if_available()

    this_dir = Path(__file__).parent
    if str(this_dir) not in sys.path:
        sys.path.append(str(this_dir))

    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    orig_stdout = sys.stdout
    tee = Tee()
    sys.stdout = tee

    drop_options = [False, True]
    df = load_data()

    try:
        for drop_flag in drop_options:
            base_dir = OUTLIER_BASE[drop_flag]
            split_results: List[pd.DataFrame] = []

            for split_key in SPLIT_STRATEGIES.keys():
                split_dir = base_dir / split_key
                plots_dir = split_dir / "plots"
                models_dir = split_dir / "models"
                plots_dir.mkdir(parents=True, exist_ok=True)
                models_dir.mkdir(parents=True, exist_ok=True)
                res_df = train_for_split(df, split_key, drop_flag, plots_dir, models_dir)
                split_results.append(res_df.assign(Split=split_key))

            if split_results:
                all_res = pd.concat(split_results, ignore_index=True)
                agg = (
                    all_res.groupby("Model")
                    .agg(
                        RMSE_mean=("RMSE", "mean"),
                        RMSE_std=("RMSE", "std"),
                        MAE_mean=("MAE", "mean"),
                        MAE_std=("MAE", "std"),
                        R2_mean=("R2", "mean"),
                        R2_std=("R2", "std"),
                        Train_RMSE_mean=("Train_RMSE", "mean") if "Train_RMSE" in all_res else ("RMSE", "mean"),
                    )
                    .reset_index()
                    .sort_values("RMSE_mean")
                )
                agg_filename = "regression_ranking_single_split.csv" if len(split_results) == 1 else "regression_ranking_aggregated_across_splits.csv"
                agg_path = base_dir / agg_filename
                agg.to_csv(agg_path, index=False)
                print(f"Saved regression ranking → {agg_path}")

            class_dir = base_dir / "classification"
            clust_dir = base_dir / "clustering"
            class_plots = class_dir / "plots"
            class_models = class_dir / "models"
            clust_plots = clust_dir / "plots"
            class_plots.mkdir(parents=True, exist_ok=True)
            class_models.mkdir(parents=True, exist_ok=True)
            clust_plots.mkdir(parents=True, exist_ok=True)

            run_classification_experiment(df, drop_flag, class_plots, class_models)
            run_clustering_experiment(df, drop_flag, clust_plots)

    finally:
        sys.stdout = orig_stdout
        with open(LOG_PATH, "w", encoding="utf-8") as f:
            f.write(tee.getvalue())
        print(f"Saved full training log → {LOG_PATH}")


if __name__ == "__main__":
    main()
