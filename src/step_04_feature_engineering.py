import numpy as np
import pandas as pd

def create_time_based_features(X):
    X["hour"] = X["time"].dt.hour
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["day_of_year"] = X["time"].dt.dayofyear
    X["month"] = X["time"].dt.month
    return X

def create_lag_features(X):
    X["T2m_rolling_6h"] = X["T2m"].rolling(6).mean()
    X["WS10m_rolling_6h"] = X["WS10m"].rolling(6).mean()
    X["T2m_rolling_6h"] = X["T2m_rolling_6h"].bfill()
    X["WS10m_rolling_6h"] = X["WS10m_rolling_6h"].bfill()
    return X

def create_sun_position_features(X):
    from pvlib import solarposition
    latitude, longitude = 47.99, 7.84 # Freiburg im Breisgau
    solar_pos = solarposition.get_solarposition(X["time"], latitude, longitude)

    # Align the index of solar_pos with X
    solar_pos.index = X.index

    X["solar_elevation"] = solar_pos["elevation"]
    X["solar_azimuth"] = solar_pos["azimuth"]

    # row 7154 is NaN for solar_elevation and solar_azimuth
    # I actually don't know why, just fillna with the mean for now
    X.fillna({"solar_elevation": X["solar_elevation"].mean(), "solar_azimuth": X["solar_azimuth"].mean()}, inplace=True)

    return X

def create_interaction_features(X):
    # Creating combined features
    X["temp_irradiation"] = X["T2m"] * X["G(i)"]
    X["wind_irradiation"] = X["WS10m"] * X["G(i)"]
    return X

def create_fourier_features(X):
    # Might help in prediction if multiple years of data are used
    X["fourier1"] = np.sin(2 * np.pi * X["day_of_year"] / 365)
    X["fourier2"] = np.cos(2 * np.pi * X["day_of_year"] / 365)
    return X
