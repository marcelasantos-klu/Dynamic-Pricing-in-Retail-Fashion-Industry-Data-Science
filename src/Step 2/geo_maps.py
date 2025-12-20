"""
City-level map generation helper for EDA.

Purpose
-------
This module generates **per-city geographic maps** of Airbnb listings to support
exploratory data analysis (EDA). Each city is visualized on an OpenStreetMap basemap,
with listings plotted by latitude/longitude and optionally colored by (log-scaled)
revenue.

Key Design Choices
------------------
- Output format: **PNG images** (static, report-friendly).
- Granularity: **one map per city** (no world/europe overview here by design).
- Performance: optional sampling to cap the number of plotted points.
- Robustness: graceful fallbacks if Plotly/Kaleido are not installed.

Typical Use
-----------
- Called from an EDA pipeline to quickly inspect spatial concentration patterns.
- Can also be run directly as a script (see `__main__` block).
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd


# -------------------------------------------------------------------
# Helper: determine a reasonable zoom level based on city extent
# -------------------------------------------------------------------
def _city_zoom(city_frame: pd.DataFrame) -> float:
    """Return a map zoom level tuned to the city's spatial extent."""
    # Compute geographic span of the city using latitude/longitude ranges
    lat_range = city_frame["lat"].max() - city_frame["lat"].min()
    lon_range = city_frame["lng"].max() - city_frame["lng"].min()
    # Use the larger of lat/lon ranges as the overall spatial extent
    span = max(lat_range, lon_range)
    # Map span thresholds to zoom levels:
    # smaller span -> stronger zoom (more detail),
    # larger span  -> weaker zoom (more context)
    if span <= 0.1:
        return 12.5
    if span <= 0.2:
        return 12
    if span <= 0.5:
        return 11
    if span <= 1:
        return 10
    if span <= 2:
        return 9
    return 8  # fallback for very spread-out areas


# -------------------------------------------------------------------
# Main API: generate one PNG map per city
# -------------------------------------------------------------------
def generate_city_maps(
    df: pd.DataFrame,
    plots_dir: Path,
    revenue_col: str = "realSum",
    city_limit: Optional[int] = None,
    log: Callable[[str], None] = print,
) -> None:
    """
    Create per-city map PNGs using plotly scatter_map.

    Parameters
    ----------
    df : pd.DataFrame
        Source dataframe containing lat/lng, City, and optionally revenue/room info.
    plots_dir : Path
        Directory where PNGs will be written.
    revenue_col : str, default "realSum"
        Column used for color scaling (log1p).
    city_limit : Optional[int], default None
        Limit the number of cities plotted; None means all.
    log : callable
        Logger function for status messages.
    """
    # Plotly is imported lazily so the rest of the pipeline
    # can run even if Plotly is not installed.
    try:
        import plotly.express as px  # type: ignore
    except Exception:
        log("Plotly not available; skipping city map exports.\n")
        return

    # Remove listings without geographic coordinates;
    # they cannot be plotted on a map.
    max_points_geo = 12000
    geo_df = df.dropna(subset=["lat", "lng"]).copy()

    # Cap the number of plotted points to keep map rendering fast
    # and avoid excessive overplotting in dense cities.
    if len(geo_df) > max_points_geo:
        geo_df = geo_df.sample(max_points_geo, random_state=42)

    # Use log-scaled revenue for coloring to reduce skewness
    # and improve visual interpretability.
    if revenue_col in geo_df.columns:
        geo_df["_log_revenue"] = np.log1p(geo_df[revenue_col].clip(lower=0))
        color_col = "_log_revenue"
        color_title = "log1p(realSum)"
        color_continuous = True
    else:
        color_col = "City" if "City" in geo_df.columns else None
        color_title = "City"
        color_continuous = False

    # City information is required for per-city maps;
    # abort early if it is missing.
    if "City" not in geo_df.columns:
        log("City column missing; skipping city map exports.\n")
        return

    # Determine which cities to plot (optionally limited to top-N by size).
    city_counts = geo_df["City"].value_counts()
    top_cities = (
        city_counts.index.tolist()
        if city_limit is None
        else city_counts.head(city_limit).index.tolist()
    )

    # Iterate over cities and generate one map per city.
    for city in top_cities:
        city_df = geo_df[geo_df["City"] == city].copy()

        # Center the map on the average city location
        # and compute an automatic zoom level.
        center_lat = float(city_df["lat"].mean())
        center_lon = float(city_df["lng"].mean())
        city_zoom = _city_zoom(city_df)

        # Create the geographic scatter plot on an OpenStreetMap basemap.
        fig_city = px.scatter_map(
            city_df,
            lat="lat",
            lon="lng",
            color=color_col if color_col else None,
            hover_data={
                revenue_col: True if revenue_col in city_df.columns else False,
                "room_type": True if "room_type" in city_df.columns else False,
                "person_capacity": True if "person_capacity" in city_df.columns else False,
                "bedrooms": True if "bedrooms" in city_df.columns else False,
            },
            title=f"Airbnb Listings – {city}",
            opacity=0.7,
            center={"lat": center_lat, "lon": center_lon},
            zoom=city_zoom,
            map_style="open-street-map",
        )
        if color_continuous:
            fig_city.update_layout(coloraxis_colorbar=dict(title=color_title))

        safe_city = (
            str(city)
            .strip()
            .lower()
            .replace(" ", "_")
            .replace("/", "_")
            .replace("\\", "_")
        )
        city_path = plots_dir / f"map_city_{safe_city}.png"

        # Export the figure as a high-resolution PNG.
        # This requires the Kaleido backend; if unavailable,
        # we skip the export but keep the pipeline running.
        try:
            fig_city.write_image(str(city_path), width=900, height=700, scale=2)
            log(f"Saved city map PNG → {city_path}")
        except Exception as e:
            log(f"Skipped PNG export for city map {city} (install kaleido to enable): {e}")

    # Clean up temporary helper columns used only for visualization.
    if "_log_revenue" in geo_df.columns:
        geo_df.drop(columns=["_log_revenue"], inplace=True, errors="ignore")

    log("Saved city-level map PNGs to EDA plots directory.\n")


if __name__ == "__main__":
    """
    Allow running this module directly to generate city map PNGs.

    Usage (from repo root):
        python3 "src/Step 2/geo_maps.py"
    """
    # Default paths when running this module as a standalone script.
    DEFAULT_DATA = Path("data/dataSetEDA/edaDataSet.csv")
    DEFAULT_PLOTS = Path("plots&models/EDA/city_maps")
    REVENUE_COL = "realSum"

    if not DEFAULT_DATA.exists():
        raise SystemExit(f"Dataset not found at {DEFAULT_DATA}")

    df = pd.read_csv(DEFAULT_DATA)
    DEFAULT_PLOTS.mkdir(parents=True, exist_ok=True)

    def _log(msg: str) -> None:
        print(msg)

    # Generate city-level maps using default settings.
    generate_city_maps(df, DEFAULT_PLOTS, revenue_col=REVENUE_COL, city_limit=None, log=_log)
