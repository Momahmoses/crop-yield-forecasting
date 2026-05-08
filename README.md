# Agricultural Crop Yield Forecasting & Farmland Monitoring

A satellite-imagery and ML-powered yield prediction platform for smallholder farmers across North-Central Nigeria. Integrates NDVI time-series, weather data, soil properties, and farm management practices to forecast crop yields per hectare per growing season.

---

## Features

- **Multi-crop support** — Maize, Sorghum, Millet, Cassava, Yam, Rice, Groundnut
- **Leave-one-season-out validation** — robust temporal generalization testing
- **Interactive Streamlit dashboard** — Yield Map, Season Trends, Custom Forecasting
- **Feature importance analysis** — understand which factors drive yield most

---

## Input Features

| Feature | Description |
|---|---|
| `ndvi_mean` | Mean NDVI from satellite imagery (Sentinel-2 / Landsat) |
| `ndvi_max` | Peak growing-season NDVI |
| `rainfall_mm` | Total growing-season rainfall (mm) |
| `temp_mean_c` / `temp_max_c` | Mean and maximum temperatures (°C) |
| `humidity_pct` | Relative humidity (%) |
| `growing_days` | Number of active growing days |
| `fertilizer_kg` | Fertilizer applied (kg/ha) |
| `soil_ph` | Soil acidity/alkalinity |
| `soil_nitrogen` | Soil nitrogen concentration (mg/kg) |
| `irrigation` | Binary: irrigated farm (1) or rainfed (0) |
| `farm_size_ha` | Farm plot area (hectares) |

---

## Project Structure

```
crop-yield-forecasting/
├── src/
│   ├── data_generator.py    # Synthetic multi-season farm dataset
│   └── model.py             # Random Forest regressor with LOO-CV
├── data/raw/                # Generated data (git-ignored)
├── assets/                  # Saved model + encoder (git-ignored)
├── streamlit_app.py         # Main Streamlit dashboard
├── requirements.txt
└── README.md
```

---

## Setup & Run

```bash
git clone https://github.com/Momahmoses/crop-yield-forecasting.git
cd crop-yield-forecasting
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Model Architecture

- **Algorithm**: `RandomForestRegressor` (scikit-learn)
- **Hyperparameters**: 300 estimators, max depth 12, 70% feature subsampling
- **Validation**: Leave-one-season-out cross-validation
- **Target**: Yield in kg per hectare

---

## Target States

Kaduna · Kano · Niger · Benue · Plateau · Kebbi · Sokoto · Zamfara

---

## Tech Stack

`Python` · `scikit-learn` · `Streamlit` · `Plotly` · `Pandas` · `NumPy`

---

## Author

**Momah Moses** — [github.com/Momahmoses](https://github.com/Momahmoses)
