
## Objective 
Overview the Airbnb bookings and crime and safety indices in those cities. We found both datasets in Kaggle: one for Airbnb bookings (sales, room type, city) and one for crime and safety indices (city and indices), therefore, the city feature will be a key that links the two datasets. 

The model is made predict the revenue of a booking based on the specific type of accommodation, city, and crime/safety conditions. 


Project tasks (aligned with the assignment guide):
1) Business and data understanding
- Define the core business question (e.g., demand forecast, price elasticity, margin optimization) and target variable(s).
- Summarize the dataset: entities, granularity, time span, and each column’s role; note any external sources you plan to add (e.g., holidays, weather, promotions).
- Run EDA to see distributions, correlations, seasonality, and how features relate to the target; identify potential segments (e.g., category, brand, collection).

2) Data preparation
- Clean data: handle missing values, duplicates, outliers, data types, and currency/units; document any rows removed.
- Encode/scale/normalize as needed; create train/validation/test splits that respect time (if applicable) and balance classes if classification.
- Engineer features relevant to pricing and demand (e.g., discount depth, time since launch, season, stock position, competitor price proxies, lag/rolling stats).

3) Modeling
- Establish a simple baseline (e.g., mean/median, naive seasonal forecast, logistic baseline) for comparison.
- Train multiple model families with parameter variation (e.g., linear/elastic net, tree-based models such as RF/GBM/XGBoost, regularized GLMs, and, if useful, a time-series or panel model). Consider an unsupervised step (clustering segments) to inform features or pricing groups.
- Track experiments and hyperparameters; compare how feature choices and model classes affect performance.

4) Evaluation
- Use suitable metrics for the task (e.g., MAE/RMSE/SMAPE for regression; accuracy/F1/ROC-AUC/PR-AUC for classification; business KPIs such as expected margin or stock-out risk).
- Apply cross-validation or time-based splits; highlight how evaluation method changes results. Include calibration checks and error analysis (by category/season/price band).
- Present performance against the baseline and between models; explain trade-offs and when each model is preferable.

5) Documentation and reflection
- Keep code commented and reproducible (scripts or notebook) with clear step ordering.
- Write up decisions, assumptions, and limitations; note issues like data leakage risks, feature availability at prediction time, and fairness/segment impacts.
- Reflect on how good the final model is, what else could be done next, and lessons learned.

Deliverables (per guide)
- Presentation (~5 minutes): problem, dataset, approach, key results, and takeaways.
- Python implementation: executable code plus required data/notebook; comments explaining each step.
- Report (about 3–4 pages per team member): cover all tasks above with plots/graphs for data, analyses, and results; include reflection on performance and next steps.
- Oral defense readiness: every team member can explain and justify choices across data prep, modeling, and evaluation.

Checklist to satisfy assessment criteria
- Problem is well framed in both domain and technical terms; dataset complexity is appropriate.
- Data is properly prepared (encodings, balancing, handling NaNs/outliers) and documented.
- Techniques and evaluation metrics fit the problem; multiple models and parameter variations tested.
- Choices are explained in the report; structure and flow are clear.
- Team can justify design decisions and discuss limitations in an oral defense.

## Setup & Dependencies
Run the project with Python 3.9+ and these key packages:
- pandas, numpy, seaborn, matplotlib
- scikit-learn, joblib
- lightgbm, catboost, xgboost

Quick install:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install pandas numpy seaborn matplotlib scikit-learn joblib lightgbm catboost xgboost
```

macOS OpenMP note (LightGBM/XGBoost): ensure `libomp.dylib` is discoverable. If Homebrew libomp is missing, you can point to the Torch-bundled copy:
```bash
export DYLD_LIBRARY_PATH="$HOME/Library/Python/3.9/lib/python/site-packages/torch/lib:${DYLD_LIBRARY_PATH}"
export DYLD_FALLBACK_LIBRARY_PATH="$HOME/Library/Python/3.9/lib/python/site-packages/torch/lib:${DYLD_FALLBACK_LIBRARY_PATH}"
```

## How to Run
1) Data prep/EDA: run the scripts under `PythonCode/Step 1`–`Step 4` and `PythonCode/Step 2/edaOnMergedDataSet.py` (headless plotting enabled by default).  
2) Modeling: use `final_models.py` (full data) or `final_models_without_outliers.py` (IQR outlier filtering). Both train log-target models, generate plots in `plots*/`, and save pipelines in `models*/`.  
3) Logs/outputs: EDA logs to `eda_terminal_output.txt`; model performance is printed to console and plots are saved to the plots folders.  

Large model artifacts are ignored via `.gitignore`; regenerate locally by re-running the modeling scripts.


## 4 Modelling

### 4.1 Pipeline Overview
The project’s primary task is **regression**: predict Airbnb booking revenue (`realSum`) from listing characteristics (e.g., `room_type`), the city (`City`), and surrounding conditions such as `Crime_Index` (and `Safety_Index`, if present). In addition, we include **classification** and **clustering** as complementary analyses to demonstrate supervised and unsupervised methods covered in the course.

All Step 4 runs are orchestrated via `final_models_switch.py`. Experiments are executed in two configurations:
- **WithOutliers**: no outlier filtering
- **Train-only outlier filtering**: outliers removed from the **TRAIN** split only using an IQR rule (the TEST split remains unchanged)

For regression, models are trained on a log-transformed target (`log1p(realSum)`) to reduce skew. Metrics are reported on the Euro scale by applying `expm1` to predictions and clipping values to be non-negative.

Outputs (plots, CSV tables, and model artifacts) are written under:
`plots&models/<WithOutliers|WithoutOutliers>/<split_key>/...`
The full console output is also stored as `plots&models/training_terminal_output.txt`.

### 4.2 Train/Test Split & Leakage Prevention
We use a reproducible **80/20 hold-out split**, stratified by `City` (seed 42). City stratification reduces the risk that a small number of large cities dominate the split and ensures that each city is represented in both training and evaluation.

To prevent **data leakage**, any learned transformation is fitted on TRAIN only and then applied unchanged to TEST:
- **Feature engineering parameters** (e.g., bin edges / quantile thresholds) are computed on TRAIN via `compute_fe_params` and reused on TEST.
- **Outlier filtering (if enabled)** is applied only on TRAIN *after* the split; the TEST set remains untouched.
- **Preprocessing** (imputation, scaling, encoding) is fitted on TRAIN and applied to TEST.

This is critical: leakage would inflate test performance and lead to unrealistic conclusions about generalization.

### 4.3 Feature Engineering & Preprocessing

#### 4.3.1 Feature Engineering
Feature engineering is applied consistently to TRAIN and TEST using TRAIN-fitted parameters:
- **Log distances**: `log_metro_dist`, `log_dist_center`
- **Ratios/flags**: `distance_ratio`, `beds_per_person`, `capacity_per_bedroom`, `capacity_gt2`, `is_studio`
- **Amenity score and luxury flag**: mean of `attr_index_norm` / `rest_index_norm`; `is_luxury` uses the TRAIN 75th percentile plus high satisfaction (≥95) and cleanliness (≥9)
- **Buckets (TRAIN-fitted edges)**: `distance_bucket` (center/mid/outer), `metro_dist_bucket` (near/mid/far), `guest_satisfaction_bucket` (ordinal bins)
- **Superhost normalization**: `host_is_superhost` mapped to 0/1

Bucketing and simple ratios make non-linear effects easier to learn and improve robustness to scale and outliers.

#### 4.3.2 Preprocessing
All models share the same preprocessing pipeline to ensure fair comparisons:
- **Numeric**: median imputation + `StandardScaler`
- **Categorical** (`room_type`, `City`, and bucket features): most-frequent imputation + one-hot encoding (`handle_unknown="ignore"`)

### 4.4 Baseline Model
A **city-mean baseline** predicts the mean TRAIN revenue per `City`, with a global TRAIN mean fallback for cities that appear only in TEST. This baseline provides a minimal benchmark that ML models should outperform.

### 4.5 Regression Models (Main Task)
Training uses `log1p(realSum)`; evaluation metrics are computed on the Euro scale after `expm1` and clipping at zero. We compare several model families (with light parameter variations):
- **Linear**: OLS, Ridge (`alpha=1.0`)
- **Trees**: Decision Tree (depth 6 / 12)
- **Ensembles**: Random Forest (200 / 500 trees)
- **Boosting**: XGBoost / LightGBM / CatBoost (when available in the environment)

Per model, we export artifacts, plots (predicted vs actual, residuals), per-city metrics, and comparison tables such as `regression_model_comparison.csv`. Train metrics are logged to help identify overfitting.

### 4.6 Crime Interpretation Add-ons (best model only)
To make the crime contribution explicit (as required by the assignment), we add two interpretability components for the **best RMSE** regression model:
- **Crime ablation**: evaluate the same model with vs. without `Crime_Index` (and `Safety_Index`, if present). Results and deltas are stored in `crime_ablation_metrics.csv`.
- **PDP for `Crime_Index`**: for the outlier-filtered run, we generate `pdp_crime_index.png` (note: PDP is shown on the model’s log-output scale).

These analyses describe model behaviour and must not be interpreted causally.

### 4.7 Classification (Complementary)
We also run a binary classification task: detect **high-revenue** bookings.
- **Label**: `high_revenue = 1` if `realSum ≥ median(realSum_train)` (threshold computed on TRAIN only)
- **Split**: 80/20 stratified by `City` (fallback to unstratified if needed)
- **Models**: Logistic Regression, Decision Tree (depth=8), Random Forest (200 trees)

Outputs include `classification_metrics.csv`, `classification_model_comparison.csv`, a comparison plot, and confusion matrices.

### 4.8 Clustering (Complementary)
For unsupervised segmentation we use **KMeans** on engineered listing/location features **excluding `realSum`** (revenue is used only post-hoc to describe clusters).
- **k tested**: 2, 3, 4
- **Diagnostics**: inertia (elbow) and silhouette score

Note on outliers: because clustering is unsupervised (no train/test split), optional IQR outlier filtering (when enabled) is applied to the **full dataset copy** for more stable cluster summaries (not train-only).

Outputs include `cluster_summary_k*.csv`, `cluster_mean_revenue_k*.png`, `kmeans_k_diagnostics.csv`, and the elbow/silhouette plots.
