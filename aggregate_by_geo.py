"""
Aggregate Airbnb listings by shared geo/room attributes, averaging selected metrics.
Grouping: room_type, room_shared, room_private, person_capacity, multi, bedrooms, lng, lat, City.
- Drops DayType
- Averages: realSum, metro_dist, attr_index, attr_index_norm, rest_index, rest_index_norm
- Adds geo_id as "{lng}_{lat}"
Output: one row per unique combination of grouping attributes.
"""
from pathlib import Path
import pandas as pd

INPUT_PATH = Path("FinalDataSet.csv")
OUTPUT_PATH = Path("FinalDataSet_grouped.csv")

# Columns used to define uniqueness for grouping
GROUP_COLS = [
    "room_type",
    "room_shared",
    "room_private",
    "person_capacity",
    "multi",
    "bedrooms",
    "lng",
    "lat",
    "City",
]

# Numeric columns to average within each group
MEAN_COLS = [
    "realSum",
    "metro_dist",
    "attr_index",
    "attr_index_norm",
    "rest_index",
    "rest_index_norm",
]

DROP_COLS = ["DayType"]


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    # Remove columns that should not appear in the aggregated output
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns], errors="ignore")

    # Validate presence of required columns
    missing_group = [c for c in GROUP_COLS if c not in df.columns]
    missing_mean = [c for c in MEAN_COLS if c not in df.columns]
    if missing_group or missing_mean:
        raise SystemExit(
            f"Missing required columns. Group missing: {missing_group}; Mean missing: {missing_mean}"
        )

    # Build aggregation dict: mean for specified numeric cols, first for the rest (non-grouping) to retain context
    other_cols = [
        c for c in df.columns if c not in GROUP_COLS + MEAN_COLS
    ]
    agg_dict = {col: "mean" for col in MEAN_COLS}
    for col in other_cols:
        agg_dict[col] = "first"

    grouped = (
        df.groupby(GROUP_COLS, dropna=False)
        .agg(agg_dict)
        .reset_index()
    )

    # Create geo_id as "{lng}_{lat}"
    grouped["geo_id"] = grouped.apply(lambda r: f"{r['lng']}_{r['lat']}", axis=1)

    grouped.to_csv(OUTPUT_PATH, index=False)
    print(f"Wrote aggregated dataset to {OUTPUT_PATH} with {len(grouped)} rows.")


if __name__ == "__main__":
    main()
