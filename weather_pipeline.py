"""
End-to-End Weather Forecasting ML Pipeline using Open-Meteo Archive API & XGBoost.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ARCHIVE_API_URL = "https://archive-api.open-meteo.com/v1/archive"

# Core daily meteorological variables from Open-Meteo (ERA5 reanalysis)
DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "temperature_2m_mean",
    "apparent_temperature_mean",
    "precipitation_sum",
    "rain_sum",
    "wind_speed_10m_max",
    "shortwave_radiation_sum",
    "et0_fao_evapotranspiration",
]


# =====================================================================
# 1. Automated Data Fetching
# =====================================================================
def fetch_historical_weather(
    latitude: float,
    longitude: float,
    start_date: str,
    end_date: str,
    timezone: str = "UTC",
) -> pd.DataFrame:
    """
    Download historical daily weather data from the Open-Meteo Archive API
    and return it as a time-indexed Pandas DataFrame.
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": ",".join(DAILY_VARIABLES),
        "timezone": timezone,
    }

    response = requests.get(ARCHIVE_API_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()

    if "daily" not in payload:
        raise ValueError(f"Unexpected API response format: {payload}")

    df = pd.DataFrame(payload["daily"])
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    return df


# =====================================================================
# 2. Data Preprocessing & Feature Engineering
# =====================================================================
def engineer_features(
    df: pd.DataFrame,
    target_col: str = "temperature_2m_mean",
    lags: Tuple[int, ...] = (1, 2, 3, 7, 14),
    rolling_windows: Tuple[int, ...] = (3, 7, 14, 30),
    is_training: bool = True,
) -> pd.DataFrame:
    """
    Clean raw meteorological observations and construct temporal features:
      - Missing-value interpolation (time-based linear + forward/backward fill)
      - Cyclical seasonal encodings (day-of-year sine/cosine)
      - Autoregressive lag features (t-1, t-2, ...)
      - Rolling window statistics (mean, std, sum)
      - Next-day target variable (y = target at t+1) when is_training=True
    """
    data = df.copy()

    # Handle missing values via linear time interpolation, then edge fill
    data = data.interpolate(method="time").ffill().bfill()

    # Diurnal temperature range (synoptic indicator of cloud cover / air mass)
    data["temp_range"] = data["temperature_2m_max"] - data["temperature_2m_min"]

    # Cyclical encoding of day-of-year so Dec 31 and Jan 1 are adjacent
    day_of_year = data.index.dayofyear
    data["doy_sin"] = np.sin(2 * np.pi * day_of_year / 365.25)
    data["doy_cos"] = np.cos(2 * np.pi * day_of_year / 365.25)

    # Autoregressive lag features across key physical variables
    lag_variables = [
        target_col,
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "shortwave_radiation_sum",
    ]
    for var in lag_variables:
        for lag in lags:
            data[f"{var}_lag_{lag}d"] = data[var].shift(lag)

    # Day-over-day temperature tendency (24-hour warming/cooling trend)
    data["temp_diff_1d"] = data[target_col] - data[target_col].shift(1)

    # Rolling window statistics to capture synoptic & seasonal regimes
    for window in rolling_windows:
        data[f"{target_col}_roll_mean_{window}d"] = (
            data[target_col].rolling(window=window).mean()
        )
        data[f"{target_col}_roll_std_{window}d"] = (
            data[target_col].rolling(window=window).std()
        )
        data[f"precip_roll_sum_{window}d"] = (
            data["precipitation_sum"].rolling(window=window).sum()
        )

    if is_training:
        # Predict tomorrow's mean temperature (t + 1)
        data["target_next_day"] = data[target_col].shift(-1)
        data = data.dropna()
    else:
        # For inference, keep the latest row after rolling/lag warmup
        FeatureCols = [c for c in data.columns if c != "target_next_day"]
        data = data.dropna(subset=FeatureCols)

    return data


# =====================================================================
# 3. Model Architecture
# =====================================================================
def build_weather_model() -> XGBRegressor:
    """
    Instantiate a regularized XGBoost Regressor tailored for tabular
    time-series forecasting.
    """
    return XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
        early_stopping_rounds=30,
    )


# =====================================================================
# 4. Training and Evaluation
# =====================================================================
def train_and_evaluate(
    df_features: pd.DataFrame,
    test_ratio: float = 0.2,
) -> Tuple[XGBRegressor, List[str], Dict[str, float]]:
    """
    Perform a chronological train/test split (never shuffle time-series data),
    fit the XGBoost model with early stopping, and evaluate MAE and RMSE.
    """
    feature_cols = [c for c in df_features.columns if c != "target_next_day"]
    X = df_features[feature_cols]
    y = df_features["target_next_day"]

    # Strict chronological split to prevent future data leakage
    split_idx = int(len(df_features) * (1 - test_ratio))
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = build_weather_model()
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))

    # Baseline comparison: Persistence forecast (tomorrow's temp = today's temp)
    persistence_preds = X_test["temperature_2m_mean"]
    persistence_mae = mean_absolute_error(y_test, persistence_preds)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "Persistence_Baseline_MAE": persistence_mae,
    }

    print("=== Model Evaluation on Chronological Test Set ===")
    print(f"Test Period       : {X_test.index.min().date()} to {X_test.index.max().date()}")
    print(f"Model MAE         : {mae:.2f} °C")
    print(f"Model RMSE        : {rmse:.2f} °C")
    print(f"Persistence MAE   : {persistence_mae:.2f} °C (naive baseline)")
    return model, feature_cols, metrics


# =====================================================================
# 5. Prediction Function
# =====================================================================
def predict_upcoming_weather(
    model: XGBRegressor,
    feature_cols: List[str],
    latitude: float,
    longitude: float,
    recent_data: Optional[pd.DataFrame] = None,
    reference_end_date: Optional[str] = None,
) -> Dict[str, float]:
    """
    Predict the next day's mean temperature (in °C) for a given location.
    If `recent_data` (at least 35 days of daily observations) is not supplied,
    it automatically fetches the most recent 60 days from the Open-Meteo API.
    """
    if recent_data is None:
        # Open-Meteo Archive API has a ~5-day reanalysis delay; default safely
        end_dt = (
            datetime.strptime(reference_end_date, "%Y-%m-%d")
            if reference_end_date
            else datetime.utcnow() - timedelta(days=5)
        )
        start_dt = end_dt - timedelta(days=60)
        recent_data = fetch_historical_weather(
            latitude=latitude,
            longitude=longitude,
            start_date=start_dt.strftime("%Y-%m-%d"),
            end_date=end_dt.strftime("%Y-%m-%d"),
        )

    processed = engineer_features(recent_data, is_training=False)
    if processed.empty:
        raise ValueError(
            "Insufficient historical rows to compute 30-day rolling features. "
            "Provide at least 35 consecutive days of data."
        )

    latest_row = processed.iloc[[-1]][feature_cols]
    latest_date = latest_row.index[0]
    forecast_date = latest_date + timedelta(days=1)

    predicted_temp = float(model.predict(latest_row)[0])
    last_observed_temp = float(recent_data.loc[latest_date, "temperature_2m_mean"])

    return {
        "latitude": latitude,
        "longitude": longitude,
        "last_observed_date": str(latest_date.date()),
        "last_observed_temp_c": round(last_observed_temp, 2),
        "forecast_date": str(forecast_date.date()),
        "predicted_mean_temp_c": round(predicted_temp, 2),
    }


# =====================================================================
# Main Execution Entrypoint
# =====================================================================
if __name__ == "__main__":
    # Example Location: New York City (40.7128° N, 74.0060° W)
    LAT, LON = 40.7128, -74.0060
    START_DATE = "2019-01-01"
    END_DATE = "2024-12-31"

    print(f"1. Fetching historical weather data ({START_DATE} to {END_DATE})...")
    raw_df = fetch_historical_weather(LAT, LON, START_DATE, END_DATE)
    print(f"   Downloaded {len(raw_df):,} daily observations.\n")

    print("2. Preprocessing data and engineering temporal features...")
    features_df = engineer_features(raw_df, is_training=True)
    print(f"   Engineered dataset shape: {features_df.shape}\n")

    print("3. Training XGBoost Regressor and evaluating performance...")
    trained_model, feature_names, eval_metrics = train_and_evaluate(features_df)

    print("\n4. Generating upcoming weather prediction...")
    forecast = predict_upcoming_weather(
        model=trained_model,
        feature_cols=feature_names,
        latitude=LAT,
        longitude=LON,
        recent_data=raw_df.tail(60),
    )
    print(
        f"   Location            : ({forecast['latitude']}, {forecast['longitude']})\n"
        f"   Last Observed Date  : {forecast['last_observed_date']} "
        f"({forecast['last_observed_temp_c']} °C)\n"
        f"   Forecast for        : {forecast['forecast_date']} -> "
        f"Predicted Mean Temp: {forecast['predicted_mean_temp_c']} °C"
    )
