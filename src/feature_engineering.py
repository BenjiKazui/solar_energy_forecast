import numpy as np

def create_time_based_features(X):
    X["hour"] = X["time"].dt.hour
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["day_of_year"] = X["time"].dt.dayofyear
    X["month"] = X["time"].dt.month
    return X

def create_lag_features(X):
    X["T2m_lag_1h"] = X["T2m"].shift(1)
    X["WS10m_lag_1h"] = X["WS10m"].shift(1)
    X["T2m_lag_24h"] = X["T2m"].shift(24)
    X["WS10m_lag_24h"] = X["WS10m"].shift(24)

    X["T2m_lag_1h"] = X["T2m_lag_1h"].bfill()
    X["WS10m_lag_1h"] = X["WS10m_lag_1h"].bfill()
    X["T2m_lag_24h"] = X["T2m_lag_24h"].bfill()
    X["WS10m_lag_24h"] = X["WS10m_lag_24h"].bfill()
    return X

def create_rolling_features(X):
    X["T2m_rolling_3h"] = X["T2m"].rolling(3).mean()
    X["WS10m_rolling_3h"] = X["WS10m"].rolling(3).mean()
    X["T2m_rolling_6h"] = X["T2m"].rolling(6).mean()
    X["WS10m_rolling_6h"] = X["WS10m"].rolling(6).mean()

    X["T2m_rolling_3h"] = X["T2m_rolling_3h"].bfill()
    X["WS10m_rolling_3h"] = X["WS10m_rolling_3h"].bfill()
    X["T2m_rolling_6h"] = X["T2m_rolling_6h"].bfill()
    X["WS10m_rolling_6h"] = X["WS10m_rolling_6h"].bfill()

def create_sun_position_features(X):
    from pvlib import solarposition
    latitude, longitude = 47.99, 7.84 # Freiburg im Breisgau
    solar_pos = solarposition.get_solarposition(X["time"], latitude, longitude)

    # Align the index of solar_pos with X
    solar_pos.index = X.index

    X["solar_elevation"] = solar_pos["elevation"]
    X["solar_azimuth"] = solar_pos["azimuth"]

    return X

def create_interaction_features(X):
    # Create combined features
    X["temp_irradiation"] = X["T2m"] * X["G(i)"]
    X["wind_irradiation"] = X["WS10m"] * X["G(i)"]
    return X

def create_fourier_features(X):
    # Might help in prediction if multiple years of data are used
    X["fourier1"] = np.sin(2 * np.pi * X["day_of_year"] / 365)
    X["fourier2"] = np.cos(2 * np.pi * X["day_of_year"] / 365)
    return X

def create_features(df, feature_names):
    if "time_based" in feature_names:
        df = create_time_based_features(df)
    if "lag" in feature_names:
        df = create_lag_features(df)
    if "rolling" in feature_names:
        df = create_rolling_features(df)
    if "sun_position" in feature_names:
        df = create_sun_position_features(df)
    if "interaction" in feature_names:
        df = create_interaction_features(df)
    if "fourier" in feature_names:
        df = create_fourier_features(df)  
    else:
        raise ValueError("Unknown feature name, use 'time_based', 'lag', 'rolling', 'sun_position', 'interaction' or 'fourier'")
    return df