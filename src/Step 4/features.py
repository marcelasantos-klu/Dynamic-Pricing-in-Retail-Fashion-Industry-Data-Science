"""
Feature engineering helpers and simple metrics utilities.

- compute_fe_params: fit FE parameters on TRAIN to avoid leakage.
- _feature_engineering_for_ml: apply shared FE (capacity ratios, log distances, buckets, amenity score).
- remove_outliers_iqr: generic IQR filter.
- safe_r2: R² that returns NaN for tiny samples (per-city metrics).
"""
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from config import RANDOM_STATE


def safe_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute R² safely (needs at least 2 samples)."""
    if len(y_true) < 2:
        return np.nan
    return r2_score(y_true, y_pred)


def compute_fe_params(df: pd.DataFrame) -> Dict[str, float]:
    """Compute feature-engineering parameters on TRAIN ONLY to avoid leakage."""
    max_center_dist = pd.to_numeric(df.get("dist"), errors="coerce").max()
    if pd.isna(max_center_dist) or max_center_dist <= 6:
        max_center_dist = 6.0001

    max_metro_dist = pd.to_numeric(df.get("metro_dist"), errors="coerce").max()
    if pd.isna(max_metro_dist) or max_metro_dist <= 2:
        max_metro_dist = 2.0001

    attr = pd.to_numeric(df.get("attr_index_norm"), errors="coerce")
    rest = pd.to_numeric(df.get("rest_index_norm"), errors="coerce")
    amenity_score = (attr + rest) / 2
    amenity_q75 = amenity_score.quantile(0.75)
    if pd.isna(amenity_q75):
        amenity_q75 = 0.0

    return {
        "max_center_dist": float(max_center_dist),
        "max_metro_dist": float(max_metro_dist),
        "amenity_q75": float(amenity_q75),
    }


def remove_outliers_iqr(df: pd.DataFrame, col: str = "realSum", k: float = 1.5) -> pd.DataFrame:
    """Remove outliers using the IQR rule."""
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame for outlier removal.")

    series = pd.to_numeric(df[col], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = series.between(lower, upper)
    return df.loc[mask].reset_index(drop=True)


def _feature_engineering_for_ml(df: pd.DataFrame, fe_params: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Apply the common feature engineering steps to a copy of the data."""
    if fe_params is None:
        fe_params = compute_fe_params(df)

    df = df.copy()

    # Capacity-related features
    df["beds_per_person"] = (df["bedrooms"] / df["person_capacity"]).replace([np.inf, -np.inf], np.nan)
    df["capacity_per_bedroom"] = df["person_capacity"] / (df["bedrooms"] + 1)
    df["capacity_gt2"] = (df["person_capacity"] >= 3).astype(int)
    df["is_studio"] = (df["bedrooms"] == 0).astype(int)

    # Distance features (log)
    df["log_metro_dist"] = np.log1p(df["metro_dist"])
    df["log_dist_center"] = np.log1p(df["dist"])
    df["distance_ratio"] = df["metro_dist"] / (df["dist"] + 1e-3)

    max_center_dist = fe_params.get("max_center_dist", None)
    if max_center_dist is None or pd.isna(max_center_dist) or max_center_dist <= 6:
        max_center_dist = 6.0001
    df["distance_bucket"] = pd.cut(
        df["dist"],
        bins=[-0.01, 2, 6, max_center_dist],
        labels=["center", "mid", "outer"],
        include_lowest=True,
    )

    # Amenity score and luxury flag
    df["amenity_score"] = (df["attr_index_norm"] + df["rest_index_norm"]) / 2

    amenity_q75 = fe_params.get("amenity_q75", None)
    if amenity_q75 is None or pd.isna(amenity_q75):
        amenity_q75 = df["amenity_score"].quantile(0.75)
    if pd.isna(amenity_q75):
        amenity_q75 = 0.0
    df["is_luxury"] = (
        (df["amenity_score"] >= amenity_q75)
        & (df["guest_satisfaction_overall"] >= 95)
        & (df["cleanliness_rating"] >= 9)
    ).astype(int)

    # Guest satisfaction bucket
    df["guest_satisfaction_bucket"] = pd.cut(
        df["guest_satisfaction_overall"],
        bins=[0, 80, 90, 95, 100],
        labels=["low", "medium", "high", "excellent"],
        include_lowest=True,
    )

    # Clean superhost flag
    super_raw = df["host_is_superhost"].astype(str).str.strip().str.lower()
    super_map = {"t": 1, "true": 1, "y": 1, "yes": 1, "1": 1}
    df["host_is_superhost"] = super_raw.map(super_map).fillna(0).astype(int)

    # Metro distance bucket
    max_dist = fe_params.get("max_metro_dist", None)
    if max_dist is None or pd.isna(max_dist) or max_dist <= 2:
        max_dist = 2.0001
    df["metro_dist_bucket"] = pd.cut(
        df["metro_dist"],
        bins=[-0.01, 0.5, 2, max_dist],
        labels=["near", "mid", "far"],
        include_lowest=True,
    )

    # Drop redundant raw columns
    df = df.drop(columns=["metro_dist", "dist", "attr_index_norm", "rest_index_norm"])
    return df
