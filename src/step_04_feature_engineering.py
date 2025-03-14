import numpy as np


#from step_03_data_preprocessing import data_preprocessing

#X, y = data_preprocessing()


#print(X)

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
    return X

def create_sun_position_features(X):
    from pvlib.solarposition import get_solarposition
    import pandas

    latitude, longitude = 47.99, 7.84 # Freiburg im Breisgau
    solar_pos = get_solarposition(X["time"], latitude, longitude)
    X["solar_elevation"] = solar_pos["elevation"]
    X["solar_azimuth"] = solar_pos["azimuth"]
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
