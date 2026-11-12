# Weather-AI: Automated Meteorological Forecasting Pipeline

An end-to-end machine learning pipeline in Python that downloads multi-year historical weather reanalysis data from the [Open-Meteo Archive API](https://open-meteo.com/en/docs/historical-weather-api), engineers physical & autoregressive temporal features, trains a regularized **XGBoost Regressor**, and generates next-day weather forecasts.

---

## Features

1. **Automated Data Fetching (`fetch_historical_weather`)**
   - Queries daily ERA5 reanalysis observations (max/min/mean 2m temperature, apparent temperature, precipitation, rain, 10m max wind speed, shortwave radiation, and reference evapotranspiration) for any latitude and longitude.
   - Returns a clean, chronologically indexed `pandas.DataFrame`.

2. **Meteorological Feature Engineering (`engineer_features`)**
   - **Missing Value Handling:** Time-based linear interpolation with forward/backward edge filling.
   - **Diurnal Temperature Range (`temp_range`):** Captures synoptic cloud-cover and air-mass characteristics.
   - **Cyclical Seasonal Encoding (`doy_sin`, `doy_cos`):** Preserves continuity across December 31 and January 1.
   - **Autoregressive Lags & Tendencies:** Multi-day lags (`1, 2, 3, 7, 14` days) and 24-hour temperature change (`temp_diff_1d`).
   - **Rolling Window Statistics:** `3, 7, 14, 30`-day rolling means, standard deviations, and cumulative precipitation.

3. **Model Architecture (`build_weather_model`)**
   - Uses an `XGBRegressor` with $L_1$/$L_2$ regularization, column/row subsampling, and early stopping to model non-linear interactions between solar radiation, wind, precipitation, and thermal inertia.

4. **Strict Chronological Evaluation (`train_and_evaluate`)**
   - Splits data strictly in time order to prevent future data leakage.
   - Evaluates **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)** against a meteorological **Persistence Baseline** ($\hat{y}_{t+1} = y_t$).

5. **Inference & Forecasting (`predict_upcoming_weather`)**
   - Accepts custom recent observations or automatically fetches the latest 60-day window for a given `(latitude, longitude)` to predict the next day's mean temperature.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/henry1056/Weather-ai.git
cd Weather-ai
pip install -r requirements.txt
```

Or install the packages directly:

```bash
pip install pandas numpy requests scikit-learn xgboost
```

---

## Usage

### Run the Full Pipeline

Execute `weather_pipeline.py` to download historical data (defaults to New York City, `40.7128° N, -74.0060° W`), train the model, print evaluation metrics, and output a next-day forecast:

```bash
python weather_pipeline.py
```

### Use as a Module for Any Location

```python
from weather_pipeline import (
    fetch_historical_weather,
    engineer_features,
    train_and_evaluate,
    predict_upcoming_weather,
)

# 1. Fetch historical data (e.g., London, UK)
lat, lon = 51.5074, -0.1278
raw_df = fetch_historical_weather(lat, lon, start_date="2019-01-01", end_date="2024-12-31")

# 2. Engineer features & train model
features_df = engineer_features(raw_df, is_training=True)
model, feature_cols, metrics = train_and_evaluate(features_df)

# 3. Predict upcoming weather (auto-fetches recent 60 days if recent_data is omitted)
forecast = predict_upcoming_weather(
    model=model,
    feature_cols=feature_cols,
    latitude=lat,
    longitude=lon,
)
print(forecast)
```

---

## Project Structure

```text
Weather-ai/
├── weather_pipeline.py   # End-to-end data fetching, feature engineering, training & inference
├── requirements.txt      # Python package dependencies
├── README.md             # Project documentation
└── LICENSE               # License file
```

## License

See [LICENSE](LICENSE) for details.