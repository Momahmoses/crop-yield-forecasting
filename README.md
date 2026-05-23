# Agricultural Crop Yield Forecasting & Farmland Monitoring

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-deployed-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Satellite-imagery and ML-powered yield prediction platform for smallholder farmers across North-Central Nigeria, integrating NDVI time-series, weather, soil properties, and farm management data to forecast crop yields per hectare per growing season.

---

## Problem Statement

Smallholder farmers in Nigeria's food belt lack access to yield forecasts, making it impossible to plan inputs, storage, or marketing. This platform provides per-farm, per-season yield forecasts to improve food security and income planning.

---

## Features

| Feature | Description |
|---------|-------------|
| Multi-Crop Support | Maize, Sorghum, Millet, Cassava, Yam, Rice, Groundnut |
| NDVI / EVI Integration | Sentinel-2 and Landsat-8 spectral indices |
| LightGBM Regression | Yield prediction with feature importance ranking |
| Leave-One-Season-Out CV | Robust temporal generalisation testing |
| Streamlit Dashboard | Yield map, season trends, custom forecasting tool |

---

## Input Features

| Feature | Source |
|---------|--------|
| `ndvi_mean` | Sentinel-2 / Landsat-8 |
| `rainfall_mm` | NIMET rainfall stations |
| `soil_quality` | FAO HarvestChoice soil data |
| `temperature_mean` | ERA5 reanalysis |
| `farm_management` | Extension officer surveys |
| `crop_type` | Satellite classification |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Remote Sensing | NDVI/EVI from Sentinel-2, Landsat-8 |
| Machine Learning | LightGBM, scikit-learn |
| Geospatial | GeoPandas, Folium |
| Dashboard | Streamlit, Plotly |
| Data | pandas, NumPy |

---

## Quick Start

```bash
git clone https://github.com/Momahmoses/crop-yield-forecasting.git
cd crop-yield-forecasting
pip install -r requirements.txt
streamlit run streamlit_app.py
```

---

## Author

**Momah Moses**, Geospatial AI Engineer & Data Scientist
[GitHub](https://github.com/Momahmoses) · [Portfolio](https://momahmoses-ng-gis-portfolio.hf.space)
