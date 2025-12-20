"""
Central configuration for Step 4 experiments.

Contains only constants/paths used across modules (data path, seeds,
split strategies, and output roots for with/without outliers).
No code side effects here; all modules import from this file.
"""
from pathlib import Path

# Paths & seeds
DATA_PATH = Path("data/FinalFile/FinalDataSet_geo_merged.csv")
RANDOM_STATE = 42
LOG_PATH = Path("plots&models/training_terminal_output.txt")

# Output roots for with/without outliers
OUTLIER_BASE = {
    False: Path("plots&models/WithOutliers"),
    True: Path("plots&models/WithoutOutliers"),
}

# Train/test split strategies (city-stratified, could do multiple seeds)
SPLIT_SEEDS = [42]
SPLIT_STRATEGIES = {
    f"city_stratified_80_20_seed{seed}": {
        "test_size": 0.2,
        "random_state": seed,
        "stratify_by_city": True,
    }
    for seed in SPLIT_SEEDS
}
DEFAULT_SPLIT_STRATEGY = "city_stratified_80_20_seed42"
