"""
Synthetic dataset generator for crop yield forecasting.
Produces time-series records per farm plot with NDVI, weather, soil, and yield.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

RANDOM_SEED = 42
N_PLOTS = 2000
SEASONS = ["2019", "2020", "2021", "2022", "2023"]

CROPS = ["Maize", "Sorghum", "Millet", "Cassava", "Yam", "Rice", "Groundnut"]
STATES = [
    ("Kaduna", 10.52, 7.44), ("Kano", 12.00, 8.52), ("Niger", 9.93, 6.56),
    ("Benue", 7.34, 8.78), ("Plateau", 9.21, 9.52), ("Kebbi", 12.45, 4.20),
    ("Sokoto", 13.06, 5.24), ("Zamfara", 12.17, 6.66),
]


def generate_crop_dataset(n_plots: int = N_PLOTS, seed: int = RANDOM_SEED) -> pd.DataFrame:
    """
    Generate synthetic crop yield dataset representative of North-Central Nigeria.

    Returns
    -------
    pd.DataFrame
        Multi-season farm-level records with features and yield_kg_per_ha.
    """
    rng = np.random.default_rng(seed)
    records = []

    for plot_id in range(n_plots):
        state, lat, lon = STATES[rng.integers(0, len(STATES))]
        crop = CROPS[rng.integers(0, len(CROPS))]
        farm_size_ha = round(float(rng.uniform(0.5, 10.0)), 2)
        soil_ph = round(float(rng.normal(6.2, 0.5)), 2)
        soil_nitrogen = round(float(rng.beta(3, 5) * 100), 2)
        irrigation = int(rng.random() < 0.3)

        for season in SEASONS:
            ndvi_mean = round(float(np.clip(rng.normal(0.55, 0.15), 0.1, 0.9)), 4)
            ndvi_max = round(float(np.clip(ndvi_mean + rng.uniform(0.05, 0.2), 0.1, 0.95)), 4)
            rainfall_mm = max(0.0, round(float(rng.normal(650, 150)), 1))
            temp_mean_c = round(float(rng.normal(28, 3)), 2)
            temp_max_c = round(float(temp_mean_c + rng.uniform(4, 8)), 2)
            humidity_pct = round(float(np.clip(rng.normal(65, 15), 20, 95)), 1)
            growing_days = int(rng.integers(90, 160))
            fertilizer_kg = round(float(rng.lognormal(3.5, 0.8)), 1)

            base_yield = {
                "Maize": 2800, "Sorghum": 1800, "Millet": 1200,
                "Cassava": 12000, "Yam": 9000, "Rice": 3200, "Groundnut": 1600,
            }[crop]

            yield_val = (
                base_yield
                * (0.4 + 0.6 * ndvi_mean)
                * (0.7 + 0.3 * min(rainfall_mm / 800, 1.2))
                * (1.0 + 0.15 * irrigation)
                * (0.85 + 0.15 * min(fertilizer_kg / 80, 1.2))
                * rng.normal(1.0, 0.08)
            )

            records.append({
                "plot_id": plot_id,
                "state": state,
                "latitude": round(lat + rng.normal(0, 0.2), 6),
                "longitude": round(lon + rng.normal(0, 0.2), 6),
                "crop": crop,
                "season": season,
                "farm_size_ha": farm_size_ha,
                "soil_ph": soil_ph,
                "soil_nitrogen": soil_nitrogen,
                "irrigation": irrigation,
                "ndvi_mean": ndvi_mean,
                "ndvi_max": ndvi_max,
                "rainfall_mm": rainfall_mm,
                "temp_mean_c": temp_mean_c,
                "temp_max_c": temp_max_c,
                "humidity_pct": humidity_pct,
                "growing_days": growing_days,
                "fertilizer_kg": fertilizer_kg,
                "yield_kg_per_ha": round(max(0, yield_val), 1),
            })

    return pd.DataFrame(records)


def save_dataset(output_dir: str | Path = "data/raw") -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = generate_crop_dataset()
    path = output_dir / "crop_data.csv"
    df.to_csv(path, index=False)
    return path


if __name__ == "__main__":
    saved = save_dataset()
    print(f"Saved to {saved}")
