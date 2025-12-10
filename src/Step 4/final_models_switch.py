"""
Train Airbnb price models and run additional experiments.

For each split strategy, we run everything twice: once keeping target outliers,
once with an IQR-based outlier filter. Per split we:
1) Predict revenue (regression),
2) Classify high vs. normal revenue,
3) Cluster similar listings.

All artifacts go to plots&models/ in split-specific folders under
"WithOutliers" and "WithoutOutliers".
"""

"""
# ===============================================================
# PROJECT OVERVIEW & DOCUMENTATION
# ===============================================================

This script implements the complete machine‑learning pipeline for the
Airbnb revenue prediction project. It follows the full Data Science
lifecycle used in the assignment:

1. **Business Understanding**
   - Goal: Predict booking revenue based on accommodation type,
     city, geographic conditions, and safety levels (Crime_Index).
   - Additional tasks: Identify high‑revenue bookings (classification),
     and uncover natural groups of listings (clustering).

2. **Data Understanding**
   - Data is loaded from the pre‑processed CSV file produced in Step 3.
   - Crime Index is standardized under the column `Crime_Index`.
   - A technical `geo_id` key is removed.

3. **Data Preparation**
   - Feature engineering creates meaningful variables:
       * capacity ratios (`beds_per_person`, `capacity_per_bedroom`)
       * log‑transformed distances (`log_metro_dist`, `log_dist_center`)
       * amenity score (`amenity_score`)
       * categorical buckets (`metro_dist_bucket`)
       * boolean conveniences (`capacity_gt2`, `is_studio`)
   - All models share the same preprocessing pipeline: imputation,
     scaling, and one‑hot encoding.

4. **Modeling**
   - Regression models: Linear Regression, Ridge, Decision Tree,
     Random Forest, XGBoost, LightGBM, CatBoost (if available).
   - A baseline model predicts the mean revenue per city.
   - Models are trained on log‑transformed revenue and evaluated on
     the natural Euro scale.

5. **Evaluation**
   - Global metrics: RMSE, MAE, R².
   - Aggregated metrics:
       * average per‑city MAE (MAE_city_mean)
       * average bucket RMSE (RMSE_bucket_mean)
   - Diagnostic plots:
       * predicted vs actual
       * residual histograms
       * residuals vs predicted
       * feature importances
       * per‑city MAE barplots
       * RMSE per price‑bucket
       * R² heatmap across city × model

6. **Classification**
   - High‑revenue classification (above median).
   - Models: Logistic Regression, Decision Tree, Random Forest.
   - Metrics: Accuracy, Precision, Recall, F1.
   - Confusion matrices saved for interpretation.

7. **Clustering**
   - K‑Means clustering with 3 clusters.
   - Features include revenue, capacity, amenities, distances,
     and Crime_Index to capture safety differences.
   - Outputs: cluster summary CSV + mean‑revenue bar plot.

8. **Reproducibility**
   - All output (plots, models, metrics) is written into
     deterministic directory structures.
   - Full terminal log is saved to `training_terminal_output.txt`.

This centralised documentation explains how each function contributes
to the end‑to‑end experimental flow and how all components satisfy the
requirements of the assignment.
"""

from __future__ import annotations

import ctypes
import os
import io
import sys
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor, plot_tree, DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans


# ===============================================================
# PATHS & CONFIGURATION
# ===============================================================

DATA_PATH = Path("data/FinalFile/FinalDataSet_geo_merged.csv")
RANDOM_STATE = 42
LOG_PATH = Path("plots&models/training_terminal_output.txt")
OUTLIER_BASE = {
    False: Path("plots&models/WithOutliers"),
    True: Path("plots&models/WithoutOutliers"),
}
# Add more split options here; all are run sequentially.
SPLIT_STRATEGIES = {
    "city_stratified_80_20_seed42": {"test_size": 0.2, "random_state": 42, "stratify_by_city": True}
   
}
DEFAULT_SPLIT_STRATEGY = "city_stratified_80_20_seed42"

# ===============================================================
# OPTIONAL LIBOMP FOR LIGHTGBM ON MACOS - EVT Raus 
# ===============================================================

def load_libomp_if_available() -> None:
    """Try to load the libomp runtime on macOS to avoid LightGBM/XGBoost errors.

    XGBoost and LightGBM depend on OpenMP (libomp) for multi-threading. On macOS
    this library is sometimes not on the default search path, which can lead to
    runtime import errors. This helper attempts to load libomp from a few common
    installation locations and sets the corresponding environment variables so
    the native libraries of the gradient-boosting frameworks can be initialized
    correctly. If nothing can be loaded, the script still runs, but some models
    may fail at import or fall back to single-threaded execution.
    """
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


# ===============================================================
# DATA LOADING
# ===============================================================

def load_data() -> pd.DataFrame:
    """Load the prepared Airbnb dataset from disk.

    This function reads the FinalDataSet_geo_merged.csv file, renames the
    crime column to `Crime_Index` (so we always use the same name in the code)
    and drops the technical `geo_id` column that was only used to merge rows.
    The returned DataFrame is the starting point for all later steps such as
    feature engineering, model training and evaluation.
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    df = pd.read_csv(DATA_PATH)
    df = df.rename(columns={"Crime Index": "Crime_Index"})
    df = df.drop(columns=["geo_id"], errors="ignore")
    return df


def make_train_test_split(
    X: pd.DataFrame,
    y_log: pd.Series,
    city_series: pd.Series,
    strategy_key: Optional[str] = None,
):
    """Create a train/test split according to a named strategy.

    All split settings are defined in the SPLIT_STRATEGIES dictionary. Each
    strategy defines:
    - how big the test set should be,
    - which random seed to use,
    - and whether the split should be stratified by city.

    Stratifying by city means that the share of each city is similar in the
    train and test sets. This is important because price levels differ across
    cities, and we want a fair evaluation.
    """
    key = strategy_key or DEFAULT_SPLIT_STRATEGY
    cfg = SPLIT_STRATEGIES.get(key)
    if cfg is None:
        print(f"Unknown split '{key}', falling back to {DEFAULT_SPLIT_STRATEGY}")
        cfg = SPLIT_STRATEGIES[DEFAULT_SPLIT_STRATEGY]
        key = DEFAULT_SPLIT_STRATEGY
    stratify = city_series if cfg.get("stratify_by_city") else None
    return train_test_split(
        X, y_log,
        test_size=cfg["test_size"],
        random_state=cfg["random_state"],
        stratify=stratify,
    )


# ===============================================================
# PREPROCESSOR
# ===============================================================

def build_preprocessor(feature_df: pd.DataFrame) -> ColumnTransformer:
    """Build a preprocessing pipeline for numeric and categorical features.

    Numeric columns are filled with the median if values are missing and then
    scaled to have mean 0 and variance 1. Categorical columns (`room_type`,
    `City`, `metro_dist_bucket`) are filled with the most frequent value and
    one-hot encoded.

    The same preprocessor is used for all models so that they are trained and
    evaluated on exactly the same transformed feature space.
    """
    categorical_features = [
        "room_type",
        "City",
        "metro_dist_bucket",
        "distance_bucket",
        "guest_satisfaction_bucket",
    ]
    numeric_features = [c for c in feature_df.columns if c not in categorical_features]

    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_features),
            ("cat", cat_pipe, categorical_features),
        ]
    )


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    return preprocessor.get_feature_names_out().tolist()


# ===============================================================
# METRICS & VISUALIZATION
# ===============================================================

def eval_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute the main regression metrics on the original Euro scale.

    The models are trained on log-transformed targets, but for reporting we
    transform predictions back to Euros. We then calculate:
    - RMSE: root mean squared error (strongly penalizes large mistakes),
    - MAE: mean absolute error (average absolute deviation in Euros),
    - R²: how much of the variance in the data is explained by the model.
    """
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def plot_pred_vs_actual(y_true, y_pred, model_name, out_path):
    """Create a scatter plot of predicted vs. actual prices on the raw Euro scale.

    Each point is one booking. The diagonal line shows perfect predictions.
    Points above this line are overpredictions, points below are underpredictions.
    We color them red or green so it is easy to see how the model behaves for
    different price ranges.

    We choose this type of plot because it gives an immediate visual answer to
    the question: "How close are our predictions to the true prices across the
    entire price spectrum?". It also makes it easy to spot systematic patterns,
    for example if the model consistently underestimates very expensive bookings.

    There are alternative plots (e.g. calibration curves, error vs. predicted
    value, or quantile–quantile plots of residuals), but this simple
    predicted-vs-actual scatter is usually the most intuitive for non-technical
    stakeholders and fits well into a short project report or slide deck.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = ["red" if yp > yt else "green" for yt, yp in zip(y_true, y_pred)]
    ax.scatter(y_true, y_pred, alpha=0.5, s=12, c=colors)
    legend_elements = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="red", markersize=6, label="Overprediction"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="green", markersize=6, label="Underprediction"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", title="Error type")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, "--", color="black", linewidth=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel("Actual price (€)")
    ax.set_ylabel("Predicted price (€)")
    ax.set_title(f"Predicted vs Actual (raw scale) – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=250, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(y_true, y_pred, model_name, out_path):
    """Plot a histogram of residuals (actual minus predicted price) on the Euro scale.

    If the model is well calibrated, the residuals are roughly centered around
    zero and fairly symmetric. Strong skew or very heavy tails indicate that
    the model systematically over- or underestimates some bookings.

    We use a simple histogram here because it shows at a glance how large the
    typical errors are and whether there are many extreme mistakes. For a more
    detailed statistical analysis one could also use a residuals-vs-fitted
    plot or a QQ-plot, but for our project the histogram gives a good balance
    between insight and simplicity.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(residuals, bins=40, alpha=0.8)
    ax.axvline(0, color="gray", linestyle="--")
    ax.set_title(f"Residuals (raw scale) – {model_name}")
    ax.set_xlabel("Actual - Predicted (Euro)")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


# ---------------------------------------------------------------
# Residuals vs Predicted Plot (diagnostic for systematic errors)
# ---------------------------------------------------------------
def plot_residuals_vs_pred(y_true, y_pred, model_name, out_path):
    """Residuals vs Predicted plot to diagnose systematic model errors.

    This plot helps reveal patterns such as:
    - increasing error with higher predicted prices (heteroskedasticity),
    - systematic under/overestimation in certain value ranges,
    - non-random structure indicating model misspecification.

    If the model is well-behaved, residuals should be scattered randomly
    around zero with no visible pattern.
    """
    residuals = y_true - y_pred
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.scatter(y_pred, residuals, alpha=0.4, s=10)
    ax.axhline(0, color="gray", linestyle="--")
    ax.set_xlabel("Predicted price (€)")
    ax.set_ylabel("Residual (Actual - Predicted, €)")
    ax.set_title(f"Residuals vs Predicted – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_feature_importance(df: pd.DataFrame, model_name: str, out_path: Path):
    """Visualize the most important features for a tree-based model.

    The input DataFrame must contain two columns: `feature` and `importance`.
    The plot shows which variables have the strongest influence on the model's
    predictions, which is very useful when explaining results to non-technical
    stakeholders.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    df_sorted = df.sort_values("importance")
    ax.barh(df_sorted["feature"], df_sorted["importance"])
    ax.set_title(f"Top {len(df)} Features – {model_name}")
    plt.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def feature_importance_df(pipeline: Pipeline, top_n: int = 10):
    """Extract the top-n feature importances from a fitted pipeline, if possible.

    Only tree-based models such as RandomForest, XGBoost, LightGBM and CatBoost
    provide a `feature_importances_` attribute. For linear models this function
    returns None. Feature names come from the preprocessor so that we can match
    them back to the original input variables.
    """
    reg = pipeline.named_steps["regressor"]
    if not hasattr(reg, "feature_importances_"):
        return None
    feature_names = get_feature_names(pipeline.named_steps["preprocessor"])
    importances = pd.Series(reg.feature_importances_, index=feature_names)
    top = importances.sort_values(ascending=False).head(top_n)
    return top.reset_index().rename(columns={"index": "feature", 0: "importance"})


# ===============================================================
# OUTLIER HANDLING
# ===============================================================

def remove_outliers_iqr(df: pd.DataFrame, col: str = "realSum", k: float = 1.5) -> pd.DataFrame:
    """Remove outliers from a column using the IQR (interquartile range) rule.

    Values below Q1 - k * IQR or above Q3 + k * IQR are treated as outliers.
    In this project we usually apply this to `realSum` to reduce the impact of
    extremely cheap or extremely expensive bookings. The function prints how
    many rows were removed and returns a cleaned copy of the DataFrame.
    """
    if col not in df.columns:
        raise KeyError(f"Column '{col}' not found in DataFrame for outlier removal.")

    series = pd.to_numeric(df[col], errors="coerce")
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = series.between(lower, upper)
    removed = (~mask).sum()
    print(
        f"Outlier removal on '{col}': "
        f"IQR={iqr:.2f}, lower={lower:.2f}, upper={upper:.2f}. "
        f"Removed {removed} of {len(df)} rows ({removed / len(df) * 100:.2f}%)."
    )

    return df.loc[mask].reset_index(drop=True)


# ===============================================================
# CLASSIFICATION & UNSUPERVISED LEARNING
# ===============================================================

def _feature_engineering_for_ml(df: pd.DataFrame) -> pd.DataFrame:
    """Apply the common feature engineering steps to a copy of the data.

    This helper is used by regression, classification and clustering so that
    all experiments work on exactly the same feature space. The idea is to
    translate raw information (e.g. number of bedrooms, distances, amenity
    indices) into more meaningful variables that are easier for the models
    to learn from and easier for us to interpret in the report.

    Examples:
    - capacity ratios show how "tight" the accommodation is filled,
    - log-distances reduce the impact of a few very large distance values,
    - the amenity score combines two indices into a single measure of how
      attractive the area around the listing is,
    - distance buckets ("near/mid/far") make it simple to talk about
      different location categories in the discussion.
    """
    df = df.copy()

    # Capacity features capture how many people share space.
    # These correlate strongly with price in Airbnb markets.
    # --- Capacity-related features: how many people per bedroom/bed and simple flags ---
    df["beds_per_person"] = (df["bedrooms"] / df["person_capacity"]).replace([np.inf, -np.inf], np.nan)
    df["capacity_per_bedroom"] = df["person_capacity"] / (df["bedrooms"] + 1)
    df["capacity_gt2"] = (df["person_capacity"] >= 3).astype(int)
    df["is_studio"] = (df["bedrooms"] == 0).astype(int)

    # Log-transform distances to reduce the impact of extreme values
    # and make the distribution more model-friendly.
    # --- Distance features on a log scale to reduce the influence of extreme values ---
    df["log_metro_dist"] = np.log1p(df["metro_dist"])
    df["log_dist_center"] = np.log1p(df["dist"])

    # Distance ratio between metro distance and distance to city centre.
    # A low ratio means "well connected relative to distance" while a high ratio
    # indicates poor public transport for a given distance from the centre.
    df["distance_ratio"] = df["metro_dist"] / (df["dist"] + 1e-3)

    # Bucket the distance to the city centre into interpretable zones:
    # city centre, mid-range zone, and outer area.
    max_center_dist = df["dist"].max()
    if pd.isna(max_center_dist) or max_center_dist <= 6:
        max_center_dist = 6.0001
    df["distance_bucket"] = pd.cut(
        df["dist"],
        bins=[-0.01, 2, 6, max_center_dist],
        labels=["center", "mid", "outer"],
        include_lowest=True,
    )

    # Combine attraction and restaurant indices into a single metric.
    # Serves as a proxy for neighbourhood quality.
    # --- Simple amenity score: average of attraction and restaurant indices ---
    df["amenity_score"] = (df["attr_index_norm"] + df["rest_index_norm"]) / 2

    # Bucket guest satisfaction into ordinal categories. The raw scores are
    # heavily concentrated in the high range, so categories are easier to
    # interpret and more robust for the models.
    df["guest_satisfaction_bucket"] = pd.cut(
        df["guest_satisfaction_overall"],
        bins=[0, 80, 90, 95, 100],
        labels=["low", "medium", "high", "excellent"],
        include_lowest=True,
    )

    # Simple "luxury" flag: high amenity score, very good guest satisfaction
    # and high cleanliness. This does NOT use the target (realSum) to avoid
    # target leakage and instead relies only on quality-related inputs.
    amenity_q75 = df["amenity_score"].quantile(0.75)
    df["is_luxury"] = (
        (df["amenity_score"] >= amenity_q75)
        & (df["guest_satisfaction_overall"] >= 95)
        & (df["cleanliness_rating"] >= 9)
    ).astype(int)

    # Normalize host_is_superhost because dataset contains mixed types (str/boolean/int).
    # --- Clean 0/1 representation for "superhost" status ---
    super_raw = df["host_is_superhost"].astype(str).str.strip().str.lower()
    super_map = {"t": 1, "true": 1, "y": 1, "yes": 1, "1": 1}
    df["host_is_superhost"] = super_raw.map(super_map).fillna(0).astype(int)

    # Convert continuous metro distance into categories that are easier to interpret.
    # --- Group metro distance into three intuitive categories: near, mid and far ---
    max_dist = df["metro_dist"].max()
    if pd.isna(max_dist) or max_dist <= 2:
        max_dist = 2.0001
    df["metro_dist_bucket"] = pd.cut(
        df["metro_dist"],
        bins=[-0.01, 0.5, 2, max_dist],
        labels=["near", "mid", "far"],
        include_lowest=True,
    )

    # Drop redundant raw distance and amenity columns now that we have engineered versions.
    df = df.drop(columns=["metro_dist", "dist", "attr_index_norm", "rest_index_norm"])

    return df


def run_classification_experiment(
    base_df: pd.DataFrame,
    drop_outliers: bool,
    plots_dir: Path,
    models_dir: Path,
) -> None:
    """Binary classification: detect high-revenue bookings.

    In the lectures we did not only cover regression, but also classification
    and the evaluation metrics precision, recall and F1-score. To mirror this
    in our project, we derive an additional classification task from the same
    Airbnb data:

    - We label each booking as "high_revenue" (1) or "not high" (0) based on
      whether its revenue is above or below the median.
    - We then train several standard classifiers (logistic regression and
      tree-based models) on the same feature set that we use for regression.
    - Finally, we evaluate these models with accuracy, precision, recall and
      F1-score and save confusion matrices as figures.

    This allows us to answer a slightly different business question:
    "Can we identify high-value bookings?" and at the same time demonstrate
    that we understood the classification concepts from the lecture.
    """
    df = base_df.copy()
    # Work on a copy so that we do not accidentally modify the original data.
    # Optionally remove extreme revenue outliers before training the classifiers.
    # This mirrors the two settings we also use for the regression models.
    if drop_outliers:
        df = remove_outliers_iqr(df, col="realSum", k=1.5)

    # Reuse the same engineered features as in the regression task so that
    # results are comparable and easier to explain.
    df = _feature_engineering_for_ml(df)

    # Create a balanced binary label using the median revenue.
    # Median ensures roughly equal class sizes → stable classifier training.
    threshold = df["realSum"].median()
    df["high_revenue"] = (df["realSum"] >= threshold).astype(int)

    # y is the class label we want to predict; X contains all input features.
    y = df["high_revenue"]
    # Keep Crime_Index so the classifier can also learn whether safety levels correlate with high revenue.
    # NOTE:
    # Crime_Index is city-level and therefore overlaps with the City feature,
    # but we intentionally keep both because:
    #   - City encodes general market-level differences between cities,
    #   - Crime_Index explicitly represents safety conditions.
    #
    # For classification, this allows the model to test whether safety levels help
    # distinguish high-revenue vs. low-revenue bookings. Tree-based models handle
    # the redundancy without issues, and Logistic Regression is regularised via the
    # preprocessing pipeline.
    X = df.drop(columns=["realSum", "high_revenue"])        

    # Create a train/test split and stratify by the label so that the share
    # of high-revenue bookings is similar in both sets.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    preprocessor = build_preprocessor(X)

    # Define a small set of classification models that were discussed in the lecture.
    # This allows us to compare a linear model (logistic regression) with tree-based models.
    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=8, random_state=RANDOM_STATE),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=200,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = []

    for name, clf in classifiers.items():
        print(f"\n--- Classification: {name} ---")
        pipe = Pipeline(
            [
                ("preprocessor", clone(preprocessor)),
                ("classifier", clf),
            ]
        )

        # Fit the classifier on the training data using the preprocessing pipeline.
        pipe.fit(X_train, y_train)
        # Predict the high/low revenue class for the unseen test bookings.
        y_pred = pipe.predict(X_test)

        # Compute the main classification metrics from the lecture:
        # accuracy (overall correctness), precision and recall for the high class,
        # and their harmonic mean (F1-score).
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        results.append(
            {
                "Model": name,
                "Accuracy": acc,
                "Precision": prec,
                "Recall": rec,
                "F1": f1,
            }
        )

        # Save the trained classifier so it can be reused without retraining.
        model_path = models_dir / f"classifier_{name.replace(' ', '_').lower()}.pkl"
        joblib.dump(pipe, model_path)
        print(f"Saved classifier → {model_path}")

        # Build and plot the confusion matrix to visualize typical errors
        # (e.g. missed high-revenue bookings vs. false alarms).
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(4, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Low/Normal", "High"])
        ax.set_yticklabels(["Low/Normal", "High"])
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_title(f"Confusion Matrix – {name}")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, cm[i, j], ha="center", va="center", color="black")
        fig.colorbar(im, ax=ax)
        plt.tight_layout()
        cm_path = plots_dir / f"confusion_matrix_{name.replace(' ', '_').lower()}.png"
        fig.savefig(cm_path, dpi=300)
        plt.close(fig)
        print(f"Saved confusion matrix → {cm_path}")

    # Collect all metrics in a table, print it to the console and save it as CSV
    # so that we can easily include the numbers in the report.
    results_df = pd.DataFrame(results).sort_values("F1", ascending=False)
    print("\n=== CLASSIFICATION PERFORMANCE (High-Revenue vs. Other) ===")
    print(results_df)
    results_path = plots_dir / "classification_metrics.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Saved classification metrics → {results_path}")


def run_clustering_experiment(
    base_df: pd.DataFrame,
    drop_outliers: bool,
    plots_dir: Path,
) -> None:
    """Unsupervised learning: group listings into clusters with K-Means.

    The lectures also covered unsupervised learning, where we do not have a
    target variable but instead try to discover structure in the data itself.
    To reflect this in our project, we use K-Means clustering on a subset of
    numerical features (price, capacity, distance and amenity score).

    The goal is to find typical "segments" of listings, for example:
    - budget apartments with low prices and simple amenities,
    - mid-range listings with average prices and good connections,
    - premium listings with high prices and very good amenities.

    Additionally, we include `Crime_Index` in the cluster analysis so that 
    differences in safety levels across cities are reflected in the resulting 
    segments. This allows us to explore whether high-revenue clusters also 
    correspond to safer or less safe areas, even though Crime_Index is a 
    city-level variable.

    We summarize each cluster by mean and median values and create a bar plot
    of mean revenue per cluster. These outputs can then be interpreted in the
    report to discuss what kind of listings tend to generate higher revenue
    and how the clusters differ.
    """
    df = base_df.copy()
    # Again, work on a copy so that the original dataset remains unchanged.
    # Optionally remove extreme revenue outliers before running the clustering.
    # This prevents a few very unusual bookings from dominating the cluster centers.
    if drop_outliers:
        df = remove_outliers_iqr(df, col="realSum", k=1.5)

    # Use the same engineered features as in the regression and classification parts.
    df = _feature_engineering_for_ml(df)

    # Include Crime_Index so clusters can also reflect differences in safety levels.
    cluster_features = [
        "realSum",
        "person_capacity",
        "bedrooms",
        "amenity_score",
        "log_metro_dist",
        "log_dist_center",
        "Crime_Index",
    ]
    X = df[cluster_features].copy()

    # Standardize all features so that K-Means is not dominated by variables
    # with larger numeric scales (for example, revenue vs. distances).
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit K-Means on standardized features.
    # K=3 chosen for interpretability rather than statistical optimality.
    kmeans = KMeans(n_clusters=3, random_state=RANDOM_STATE, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    df["cluster"] = clusters

    # Summarize each cluster by mean and median values; this table is useful
    # for describing the typical characteristics of each segment.
    cluster_summary = (
        df.groupby("cluster")[cluster_features]
        .agg(["mean", "median"])
    )
    summary_path = plots_dir / "cluster_summary.csv"
    cluster_summary.to_csv(summary_path)
    print(f"Saved cluster summary → {summary_path}")

    # Compute and plot the average revenue in each cluster to see how the groups differ.
    # This makes it easy to talk about which cluster is the most valuable on average.
    mean_revenue = df.groupby("cluster")["realSum"].mean()
    fig, ax = plt.subplots(figsize=(5, 4))
    mean_revenue.plot(kind="bar", ax=ax)
    ax.set_title("Mean Revenue by Cluster")
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Mean realSum (€)")
    plt.tight_layout()
    plot_path = plots_dir / "cluster_mean_revenue.png"
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)
    print(f"Saved cluster revenue plot → {plot_path}")


# ===============================================================
# MAIN
# ===============================================================

def train_for_split(
    base_df: pd.DataFrame,
    split_strategy: str,
    drop_outliers: bool,
    plots_dir: Path,
    models_dir: Path,
) -> None:
    """Train and evaluate all regression models for a given split strategy.

    This function runs the complete workflow for one specific train/test split.
    The goal is always the same: predict the booking revenue as accurately as
    possible and then compare several models in a fair way.

    Concretely, for the chosen split strategy it:
    1. (Optionally) removes extreme outliers in the target variable `realSum`.
    2. Applies the common feature engineering routine so that we work with
       the same derived variables in all experiments.
    3. Splits the data into training and test sets according to the strategy
       defined in SPLIT_STRATEGIES.
    4. Trains all configured regression models (linear, tree-based and, if
       available, gradient-boosting models).
    5. Evaluates them on the test set using RMSE, MAE and R² and saves:
       - the fitted models,
       - diagnostic plots,
       - and a comparison table for the report.

    In other words: this function encapsulates one full "experiment" setup for
    a single way of splitting the data.
    """
    # Optional imports for gradient-boosting models. If these libraries are not
    # installed on the system, we simply skip them and still run the rest.
    try:
        from lightgbm import LGBMRegressor
    except ImportError:
        LGBMRegressor = None

    try:
        from catboost import CatBoostRegressor
    except ImportError:
        CatBoostRegressor = None

    try:
        from xgboost import XGBRegressor
    except ImportError:
        XGBRegressor = None

    # Start from a fresh copy of the full dataset for this particular configuration.
    df = base_df.copy()

    # Optionally remove extreme revenue outliers so that very unusual bookings
    # do not dominate the model fit or the error metrics.
    if drop_outliers:
        df = remove_outliers_iqr(df, col="realSum", k=1.5)

    # Apply the shared feature engineering so that all models see the same
    # transformed feature space (capacity ratios, log distances, etc.).
    df = _feature_engineering_for_ml(df)

    # Define the target variable: the booking revenue in Euros.
    y_raw = df["realSum"]
    # Work with the log-transformed revenue during training to reduce skewness
    # in the target distribution and make the regression problem easier.
    y_log = np.log1p(y_raw)
    # Keep Crime_Index as a regular numeric feature so the models can learn potential safety effects.
    X = df.drop(columns=["realSum"])

    # Print a header so it is clear in the log which configuration is currently running.
    print(f"\n=== Running split: {split_strategy} | Outlier removal: {'ON' if drop_outliers else 'OFF'} ===")
    # Create the actual train/test split according to the chosen strategy
    # (for example 80/20 split, stratified by city).
    X_train, X_test, y_train_log, y_test_log = make_train_test_split(
        X, y_log, df["City"], strategy_key=split_strategy
    )

    # Transform the log-targets back to Euros for later evaluation and plotting.
    y_train_raw = np.expm1(y_train_log)
    y_test_raw = np.expm1(y_test_log)

    # Build the preprocessing pipeline (imputation, scaling, one-hot encoding)
    # once and clone it for each model so that preprocessing is identical.
    preprocessor = build_preprocessor(X)

    print(f"Loaded {len(df):,} rows.")
    print(f"Train: {len(X_train):,}, Test: {len(X_test):,}")

    # Define a dictionary of all regression models that we want to compare.
    # Each entry is a small function that creates a fresh model instance.
    model_factories = {
        "Linear Regression (OLS)": lambda: LinearRegression(),
        "Ridge Regression": lambda: Ridge(alpha=1.0),
        "Decision Tree": lambda: DecisionTreeRegressor(max_depth=12, random_state=RANDOM_STATE),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1
        ),
    }

    if XGBRegressor is not None:
        model_factories["XGBoost"] = lambda: XGBRegressor(
            n_estimators=500,
            learning_rate=0.05,
            max_depth=8,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1,
            reg_lambda=1,
            random_state=RANDOM_STATE,
            objective="reg:squarederror",
            tree_method="hist",
        )

    if LGBMRegressor is not None:
        model_factories["LightGBM"] = lambda: LGBMRegressor(
            n_estimators=400,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            objective="regression",
            n_jobs=-1,
        )

    if CatBoostRegressor is not None:
        model_factories["CatBoost"] = lambda: CatBoostRegressor(
            n_estimators=400,
            learning_rate=0.05,
            depth=8,
            loss_function="RMSE",
            verbose=False,
            random_seed=RANDOM_STATE,
        )

    # ---- BASELINE EVALUATION ----
    # The baseline predicts the mean revenue per city.
    # It serves as a reference point for comparing model skill.
    # Baseline model: predict mean train revenue per city
    # This provides a simple, interpretable reference model that
    # we can compare all learned models against. It predicts for
    # each booking the average revenue of its city in the training
    # data (and falls back to the global mean if a city appears
    # only in the test set).
    baseline_name = "City-mean Baseline"
    global_mean = y_train_raw.mean()
    city_means = y_train_raw.groupby(X_train["City"]).mean()

    # Map each test booking to the mean revenue of its city
    baseline_preds = X_test["City"].map(city_means).fillna(global_mean).to_numpy()

    # Compute metrics for the baseline on the raw Euro scale
    baseline_metrics = eval_metrics(y_test_raw, baseline_preds)

    # Attach true and predicted values to the test set to analyze
    # baseline performance per city and per price bucket.
    baseline_test_df = X_test.copy()
    baseline_test_df["y_true"] = y_test_raw
    baseline_test_df["y_pred"] = baseline_preds

    # Per-city metrics for the baseline
    baseline_city_scores = (
        baseline_test_df.groupby("City")
        .apply(
            lambda g: pd.Series({
                "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
                "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                "R2": r2_score(g["y_true"], g["y_pred"]),
            }),
            include_groups=False,
        )
    )
    print(f"\nPer-City Performance – {baseline_name}")
    print(baseline_city_scores.sort_values("MAE"))

    # Plot MAE per City for the baseline
    fig, ax = plt.subplots(figsize=(7, 4))
    baseline_city_scores["MAE"].sort_values().plot(kind="bar", ax=ax)
    ax.set_title(f"MAE by City – {baseline_name}")
    ax.set_ylabel("MAE (Euro)")
    ax.set_xlabel("City")
    plt.tight_layout()
    baseline_city_mae_path = plots_dir / f"mae_by_city_{baseline_name.replace(' ', '_').lower()}.png"
    fig.savefig(baseline_city_mae_path, dpi=300)
    plt.close(fig)
    print(f"Saved MAE-by-city plot → {baseline_city_mae_path}")

    # RMSE per price bucket for the baseline
    price_bins = [0, 50, 100, 200, 500, np.inf]
    price_labels = ["0–50", "50–100", "100–200", "200–500", "500+"]

    baseline_test_df["price_bucket"] = pd.cut(
        baseline_test_df["y_true"], bins=price_bins, labels=price_labels, include_lowest=True
    )
    baseline_bucket_rmse = (
        baseline_test_df.groupby("price_bucket", observed=True)
        .apply(
            lambda g: np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
            include_groups=False,
        )
    )

    # Prepare the first row of the comparison table, including
    # global metrics and aggregates over cities and price buckets.
    baseline_row = {
        "Model": baseline_name,
        **baseline_metrics,
        "MAE_city_mean": baseline_city_scores["MAE"].mean(),
        "RMSE_bucket_mean": baseline_bucket_rmse.mean(),
    }
    results = [baseline_row]

    fig, ax = plt.subplots(figsize=(7, 4))
    baseline_bucket_rmse.plot(kind="bar", ax=ax)
    ax.set_title(f"RMSE by Price Bucket – {baseline_name}")
    ax.set_ylabel("RMSE (Euro)")
    ax.set_xlabel("Price Bucket (€)")
    plt.tight_layout()
    baseline_bucket_rmse_path = plots_dir / f"rmse_by_price_bucket_{baseline_name.replace(' ', '_').lower()}.png"
    fig.savefig(baseline_bucket_rmse_path, dpi=300)
    plt.close(fig)
    print(f"Saved RMSE-by-price-bucket plot → {baseline_bucket_rmse_path}")

    all_city_scores: Dict[str, pd.DataFrame] = {}

    for model_name, factory in model_factories.items():
        print(f"\n--- Training {model_name} ---")
        # Create a new instance of the current regression model.
        reg = factory()

        # Combine preprocessing and the regression model into one pipeline so that
        # we do not have to manually apply the transformers each time.
        pipe = Pipeline([
            ("preprocessor", clone(preprocessor)),
            ("regressor", reg),
        ])

        # Fit the model on the training data (with log-transformed revenue).
        pipe.fit(X_train, y_train_log)

        # Predict log-revenue for the test set and then transform predictions back to Euros.
        preds_log = pipe.predict(X_test)
        preds_log = np.clip(preds_log, 0, None)
        preds_raw = np.expm1(preds_log)

        # Evaluate the model on the Euro scale using RMSE, MAE and R².
        metrics = eval_metrics(y_test_raw, preds_raw)

        # Save the fitted pipeline (preprocessing + model) so we can reuse it later
        # without retraining.
        model_path = models_dir / f"model_{model_name.replace(' ', '_').lower()}.pkl"
        joblib.dump(pipe, model_path)
        print(f"Saved model → {model_path}")

        # Generate standard diagnostic plots:
        plot_pred_vs_actual(
            y_test_raw, preds_raw, model_name,
            plots_dir / f"pred_vs_actual_{model_name.replace(' ', '_').lower()}.png"
        )

        plot_residuals(
            y_test_raw, preds_raw, model_name,
            plots_dir / f"residuals_{model_name.replace(' ', '_').lower()}.png"
        )

        # Additional diagnostic: residuals vs predicted values
        plot_residuals_vs_pred(
            y_test_raw, preds_raw, model_name,
            plots_dir / f"residuals_vs_pred_{model_name.replace(' ', '_').lower()}.png"
        )

        # For tree-based models we can also extract and visualize feature importances.
        fi = feature_importance_df(pipe, top_n=10)
        if fi is not None:
            fi_path = plots_dir / f"feature_importance_{model_name.replace(' ', '_').lower()}.png"
            plot_feature_importance(fi, model_name, fi_path)
            print(f"Top 10 Features for {model_name}:\n{fi}")

        # Compute per‑city error metrics to detect market‑specific weaknesses.
        # Some cities systematically show higher errors (e.g., volatile markets).
        # Attach true and predicted values to the test set to analyze performance per city.
        test_df = X_test.copy()
        test_df["y_true"] = y_test_raw
        test_df["y_pred"] = preds_raw

        # Compute error metrics separately for each city to see where the model
        # performs better or worse.
        city_scores = (
            test_df.groupby("City")
            .apply(
                lambda g: pd.Series({
                    "MAE": mean_absolute_error(g["y_true"], g["y_pred"]),
                    "RMSE": np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                    "R2": r2_score(g["y_true"], g["y_pred"]),
                }),
                include_groups=False,
            )
        )

        print(f"\nPer-City Performance – {model_name}")
        print(city_scores.sort_values("MAE"))

        all_city_scores[model_name] = city_scores

        # Plot MAE per City for business-focused insight
        fig, ax = plt.subplots(figsize=(7, 4))
        city_scores["MAE"].sort_values().plot(kind="bar", ax=ax)
        ax.set_title(f"MAE by City – {model_name}")
        ax.set_ylabel("MAE (Euro)")
        ax.set_xlabel("City")
        plt.tight_layout()
        city_mae_path = plots_dir / f"mae_by_city_{model_name.replace(' ', '_').lower()}.png"
        fig.savefig(city_mae_path, dpi=300)
        plt.close(fig)
        print(f"Saved MAE-by-city plot → {city_mae_path}")

        # Segment revenue into business‑aligned buckets.
        # This helps determine if the model struggles more with cheap or expensive listings.
        # Create price buckets (business-oriented segmentation)
        price_bins = [0, 50, 100, 200, 500, np.inf]
        price_labels = ["0–50", "50–100", "100–200", "200–500", "500+"]

        test_df["price_bucket"] = pd.cut(test_df["y_true"], bins=price_bins, labels=price_labels, include_lowest=True)

        bucket_rmse = (
            test_df.groupby("price_bucket", observed=True)
            .apply(
                lambda g: np.sqrt(mean_squared_error(g["y_true"], g["y_pred"])),
                include_groups=False,
            )
        )

        # Add global and aggregated diagnostics so the final ranking table
        # reflects not only overall error but also fairness across cities
        # and stability across revenue levels.
        # Add this model to the comparison table, including
        # global metrics plus aggregated city- and bucket-level metrics.
        row = {
            "Model": model_name,
            **metrics,
            "MAE_city_mean": city_scores["MAE"].mean(),
            "RMSE_bucket_mean": bucket_rmse.mean(),
        }
        results.append(row)

        # Plot RMSE per price bucket
        fig, ax = plt.subplots(figsize=(7, 4))
        bucket_rmse.plot(kind="bar", ax=ax)
        ax.set_title(f"RMSE by Price Bucket – {model_name}")
        ax.set_ylabel("RMSE (Euro)")
        ax.set_xlabel("Price Bucket (€)")
        plt.tight_layout()
        bucket_rmse_path = plots_dir / f"rmse_by_price_bucket_{model_name.replace(' ', '_').lower()}.png"
        fig.savefig(bucket_rmse_path, dpi=300)
        plt.close(fig)
        print(f"Saved RMSE-by-price-bucket plot → {bucket_rmse_path}")

        # For the decision tree model, additionally save a visualization of the top
        # levels of the tree so that we can explain its decision logic.
        if model_name == "Decision Tree":
            fig, ax = plt.subplots(figsize=(20, 10))
            feature_names = get_feature_names(pipe.named_steps["preprocessor"])
            plot_tree(
                pipe.named_steps["regressor"],
                feature_names=feature_names,
                filled=True,
                max_depth=3,
                ax=ax,
            )
            tree_path = plots_dir / "decision_tree_structure.png"
            fig.savefig(tree_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved decision tree visualization → {tree_path}")

    # Collect the overall metrics of all models in a single table for easy comparison.
    results_df = pd.DataFrame(results).sort_values("RMSE").reset_index(drop=True)
    print("\n=== FINAL MODEL PERFORMANCE (RAW SCALE) ===")
    print(results_df)

    # Create a compact overview plot that compares RMSE, MAE and R² across all models.
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    results_df.plot(x="Model", y="RMSE", kind="bar", ax=axes[0], legend=False)
    axes[0].set_title("RMSE by Model")
    axes[0].set_ylabel("RMSE")
    axes[0].tick_params(axis="x", rotation=45)

    results_df.plot(x="Model", y="MAE", kind="bar", ax=axes[1], legend=False)
    axes[1].set_title("MAE by Model")
    axes[1].set_ylabel("MAE")
    axes[1].tick_params(axis="x", rotation=45)

    results_df.plot(x="Model", y="R2", kind="bar", ax=axes[2], legend=False)
    axes[2].set_title("$R^2$ by Model")
    axes[2].set_ylabel("$R^2$")
    axes[2].tick_params(axis="x", rotation=45)

    plt.tight_layout()
    overall_path = plots_dir / "model_comparison_overall.png"
    fig.savefig(overall_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved overall model comparison plot → {overall_path}")

    # Additionally build a heatmap of R² by city and model, if we have per-city scores.
    if all_city_scores:
        r2_frames = []
        for model_name, scores in all_city_scores.items():
            if "R2" in scores.columns:
                s = scores["R2"].rename(model_name)
                r2_frames.append(s)

        if r2_frames:
            r2_df = pd.concat(r2_frames, axis=1)

            fig, ax = plt.subplots(figsize=(8, 6))
            im = ax.imshow(r2_df.values, aspect="auto")

            ax.set_xticks(range(len(r2_df.columns)))
            ax.set_xticklabels(r2_df.columns, rotation=45, ha="right")
            ax.set_yticks(range(len(r2_df.index)))
            ax.set_yticklabels(r2_df.index)
            ax.set_title("$R^2$ by City and Model")

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("$R^2$")

            plt.tight_layout()
            heatmap_path = plots_dir / "r2_heatmap_city_model.png"
            fig.savefig(heatmap_path, dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"Saved R^2 heatmap by city and model → {heatmap_path}")


def main() -> None:
    """Run all experiments for every split and both outlier settings.

    For each outlier setting (with and without outliers) and for each defined
    split strategy, this function:
    - trains and evaluates the regression models,
    - then runs the classification experiment,
    - and finally runs the clustering experiment.

    All console output is captured and saved to a log file so that the full
    training history can be reviewed later. This makes it easy to reproduce
    results and to copy numbers or messages into the written report.
    """
    # On macOS, try to load the OpenMP library needed by some gradient-boosting models.
    load_libomp_if_available()

    # The Tee class forwards everything that is printed both to the real terminal
    # and to an in-memory string buffer. This lets us save a complete log later on.
    class Tee(io.StringIO):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            # Remember the original standard output so we can still print to the console.
            self._stdout = sys.stdout

        def write(self, s):
            # First write the text to the real terminal ...
            self._stdout.write(s)
            # ... and then also store it in the internal buffer.
            return super().write(s)

        def flush(self):
            self._stdout.flush()
            return super().flush()

    # Ensure that the directory for the log file exists.
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Temporarily replace sys.stdout so that all prints go through our Tee object.
    orig_stdout = sys.stdout
    tee = Tee()
    sys.stdout = tee

    # We run the full pipeline twice: once keeping all outliers, once with outlier removal.
    drop_options = [False, True]
    # Load the full Airbnb dataset once and reuse it for all runs.
    df = load_data()

    try:
        # Loop over the two outlier settings (OFF and ON).
        for drop_flag in drop_options:
            # Choose the base output folder depending on the outlier setting.
            base_dir = OUTLIER_BASE[drop_flag]
            # For each configured split strategy, train and evaluate all regression models.
            for split_key in SPLIT_STRATEGIES.keys():
                # Create subfolders for plots and model files for this specific split strategy.
                split_dir = base_dir / split_key
                plots_dir = split_dir / "plots"
                models_dir = split_dir / "models"
                plots_dir.mkdir(parents=True, exist_ok=True)
                models_dir.mkdir(parents=True, exist_ok=True)
                train_for_split(df, split_key, drop_flag, plots_dir, models_dir)

            # Once all regression runs for this outlier setting are finished, also run
            # the classification and clustering experiments exactly once.
            class_dir = base_dir / "classification"
            clust_dir = base_dir / "clustering"

            class_plots_dir = class_dir / "plots"
            class_models_dir = class_dir / "models"
            clust_plots_dir = clust_dir / "plots"

            class_plots_dir.mkdir(parents=True, exist_ok=True)
            class_models_dir.mkdir(parents=True, exist_ok=True)
            clust_plots_dir.mkdir(parents=True, exist_ok=True)

            run_classification_experiment(df, drop_flag, class_plots_dir, class_models_dir)
            run_clustering_experiment(df, drop_flag, clust_plots_dir)
    finally:
        # Restore normal printing behaviour and write everything captured by Tee to disk.
        sys.stdout = orig_stdout
        with LOG_PATH.open("w", encoding="utf-8") as f:
            f.write(tee.getvalue())
        # Inform the user where the log file was saved so they can inspect the full run later.
        print(f"Saved terminal output to {LOG_PATH}")


if __name__ == "__main__":
    main()
