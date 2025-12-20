## 4 Modelling

### 4.1 Pipeline overview
We run one consistent modelling pipeline with **regression as the main task** and **classification + clustering as complementary analyses**. The regression goal is to predict booking revenue `realSum` from accommodation attributes (e.g., `room_type`, capacity-related signals), city/location information (including engineered distance and satisfaction buckets), and crime/safety context (`Crime_Index`, and `Safety_Index` if available). Crime features are included because the project explicitly asks whether they add **predictive value beyond city and location**.

To test robustness to extreme revenues, we execute two settings: (1) no outlier filtering and (2) **train-only** outlier filtering using an IQR rule (applied on TRAIN only; TEST stays unchanged). Because `realSum` is skewed, regression models train on `log1p(realSum)` and are evaluated back on the Euro scale using `expm1`, with negative predictions clipped to zero.

### 4.2 Train/test split and leakage prevention
We use a reproducible **80/20 train/test split**, stratified by `City` when feasible (fallback to a non-stratified split if stratification is not possible). The 80/20 choice balances two needs: enough training data for more complex models and a sufficiently large test set for a stable estimate of performance on unseen bookings.

To keep evaluation realistic, we avoid data leakage by fitting all data-dependent steps on **TRAIN only** and applying the same rules to TEST. This includes feature-engineering thresholds, preprocessing (imputation/scaling/one-hot encoding), and—when enabled—outlier filtering on TRAIN only while leaving TEST untouched.

### 4.3 Feature engineering and preprocessing

#### 4.3.1 Feature engineering
We create a small set of robust engineered features to capture common non-linear patterns in revenue (e.g., “central vs. outer”, “basic vs. premium”):
- **Log distances** smooth extreme distance values.
- **Ratios and flags** summarize layout and capacity characteristics.
- **Amenity and “luxury” signals** combine amenity indices and flag high-end listings using train-based thresholds.
- **Buckets** turn continuous variables (distance, metro distance, satisfaction) into interpretable categories.
- **Cleaning** standardizes fields such as `host_is_superhost` into a consistent 0/1 format.

After engineering, raw distance and amenity-index columns are dropped to reduce redundancy and avoid feeding highly correlated duplicates.

#### 4.3.2 Preprocessing
To ensure a fair comparison across models, every regression model uses the same preprocessing pipeline:
- **Numeric features:** median imputation + standard scaling  
- **Categorical features** (including `City`, `room_type`, and the bucket variables): most-frequent imputation + one-hot encoding (with safe handling of unseen categories)

This ensures performance differences come from the model choice, not from different data handling.

### 4.4 Baseline model
We use a **city-mean baseline** that predicts the mean TRAIN revenue per `City` (with a global mean fallback). This is a strong reference because city already explains a large part of revenue differences. Any ML model must beat this baseline to justify added complexity—especially for the crime question.

### 4.5 Regression models (main task)
We compare model families from interpretable baselines to flexible non-linear learners. This matches the project goal: achieve good predictive accuracy while still being able to discuss the role of crime/safety beyond city and location. All models share the same split strategy, feature engineering, and preprocessing.

**City-mean baseline.**  
Why included: a strong real-world reference based on city averages.  
Limit: ignores within-city variation and listing-level effects.

**OLS (Linear Regression).**  
Why included: transparent additive benchmark for city, listing features, and crime/safety.  
What it shows: whether crime has a consistent linear association in an additive model.  
Limit: limited non-linear capacity and sensitive to correlated inputs.

**Ridge Regression.**  
Why included: stabilizes the linear model under many one-hot features and correlated signals (e.g., city vs crime/safety).  
What it shows: whether regularization improves generalization while staying interpretable.  
Limit: still linear; interactions are only captured indirectly via engineered features.

**Decision Tree (depth variation).**  
Why included: captures threshold-like effects naturally; depth variation illustrates under/overfitting behaviour.  
What it shows: whether non-linear splits involving location and crime features reduce error.  
Limit: single trees are unstable (high variance).

**Random Forest (tree-count variation).**  
Why included: more stable non-linear model by averaging many trees; tree-count variation tests diminishing returns.  
What it shows: whether non-linear patterns generalize better than a single tree.  
Limit: less interpretable; gains can plateau.

**Boosting (XGBoost / LightGBM / CatBoost if available).**  
Why included: strong performers on tabular data and good at capturing subtle interactions (e.g., crime × location). We include multiple boosting implementations as complementary benchmarks.  
What it shows: whether more expressive interaction modelling yields measurable improvements beyond forests and linear baselines.  
Limit: higher complexity and overfitting risk; therefore validated strictly on held-out test performance. (In this pipeline, all boosting models are trained on the same one-hot encoded feature space as the other models.)

**Selection rule.** We choose the final regression model mainly by **lowest TEST RMSE**, with MAE and R² as secondary checks, and the city-mean baseline as the minimum acceptable reference. Train metrics are logged to manually inspect train–test gaps for overfitting. Crime effects are interpreted as predictive signals, not causal effects.

### 4.6 Making the crime contribution explicit (best model only)
Because the project focuses on crime, we add two prediction-focused analyses for the **best regression model**: (i) an ablation check (best model with vs. without `Crime_Index`/`Safety_Index`) and (ii) a partial dependence plot (PDP) for `Crime_Index` to visualize the direction of the model’s dependency on crime. Both are descriptive and support interpretation; they are not causal claims.

### 4.7 Classification (complementary)
Besides predicting the exact revenue, we also run a **high-revenue classification** task (“premium vs. non-premium”). This supports the project narrative because some use cases only need a reliable high-value signal rather than an exact Euro prediction.

The label is defined using TRAIN only (median threshold), we use the same split strategy (80/20, city-stratified when feasible), and we apply train-fitted feature engineering and preprocessing to avoid leakage. We compare Logistic Regression (simple baseline), a Decision Tree (non-linear rules), and a Random Forest (more stable ensemble). This section is supportive and does not drive the final regression model choice.

### 4.8 Clustering (complementary)
We use **K-Means** clustering to group listings into segments based on listing and location features. `realSum` is excluded from clustering input by design; revenue is used only afterwards to describe clusters. We test k = 2, 3, 4 and assess separation with inertia (elbow) and silhouette. Clustering is exploratory and supports interpretation, but it is not used to choose the final regression model.
## 5 Evaluation

### 5.1 Metrics and reporting scale (what we report) – metrics from `models_regression.py`
Regression metrics are on the Euro scale after reversing the log transform: RMSE (penalizes large errors), MAE (robust average error), and R² (explained variance). We also report city‑level metrics to see whether performance is consistent across locations—a key aspect for a city/crime question.

### 5.2 Overall model comparison and selection (how we decide) – orchestrator, `models_regression.py`
All regression models share the same split/FE/preprocessing. We select the final model by lowest TEST RMSE; MAE and R² are secondary checks. The city‑mean baseline is the minimum acceptable reference. Outputs: model comparison table (`regression_model_comparison.csv`) and a horizontal bar chart (`model_comparison_overall.png`); the single‑split summary is `regression_ranking_single_split.csv`.

### 5.3 City-wise error analysis (why we do it) – `models_regression.py`
Because `City` is central, we report per‑city metrics. This reveals whether the model generalizes similarly across locations or if some cities remain harder to predict (e.g., fewer observations or different market/crime dynamics).

### 5.4 Diagnostics for the final model (what we check) – `models_regression.py`
For the selected regression model, we inspect predicted‑vs‑actual and residual plots to detect bias or heavy‑tailed errors. Diagnostics for other models are exported but not emphasized to avoid clutter; the focus is on the final pick.

### 5.5 Crime-focused evaluation (how we answer the crime part) – `models_regression.py`
We explicitly include (a) the ablation result (with vs. without `Crime_Index`/`Safety_Index`) and (b) the PDP for `Crime_Index`. If the ablation effect is small and the PDP is flat, the correct reading is that crime/safety adds limited incremental predictive power beyond city/listing features in this setup. This is a predictive finding, not a causal claim.


