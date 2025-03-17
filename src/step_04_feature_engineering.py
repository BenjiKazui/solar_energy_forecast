import numpy as np
import pandas as pd

#from step_03_data_preprocessing import data_preprocessing

#X, y = data_preprocessing()

#pd.set_option("display.max_rows", None)  # Show all rows
#pd.set_option("display.max_columns", None)  # Show all columns
#pd.set_option("display.width", 1000)  # Set max width to prevent wrapping
#pd.set_option("display.max_colwidth", None)  # Don't truncate cell content
#print(X)

def create_time_based_features(X):
    X = X.copy()
    X["hour"] = X["time"].dt.hour
    X["hour_sin"] = np.sin(2 * np.pi * X["hour"] / 24)
    X["hour_cos"] = np.cos(2 * np.pi * X["hour"] / 24)
    X["day_of_year"] = X["time"].dt.dayofyear
    X["month"] = X["time"].dt.month
    return X

def create_lag_features(X):
    X = X.copy()
    X["T2m_rolling_6h"] = X["T2m"].rolling(6).mean()
    X["WS10m_rolling_6h"] = X["WS10m"].rolling(6).mean()
    X["T2m_rolling_6h"] = X["T2m_rolling_6h"].bfill()
    X["WS10m_rolling_6h"] = X["WS10m_rolling_6h"].bfill()
    return X

def create_sun_position_features(X):
    X = X.copy()
    from pvlib import solarposition
    #print("AAAAAAAAAAAA")
    #print(X["time"][:10])
    #X["time"] = pd.to_datetime(X["time"], utc=True)utc=True)
    #print(X["time"][:10])
    #print("BBBBBBBBBBBB")
    latitude, longitude = 47.99, 7.84 # Freiburg im Breisgau
    solar_pos = solarposition.get_solarposition(X["time"], latitude, longitude)
    #print(solar_pos.dtypes)
    #print(solar_pos[["elevation", "azimuth"]].isna().sum())

    #print(X.index.duplicated().sum())

    # Ensure X does not have duplicate indices
    #if X.index.duplicated().any():
    #    X = X[~X.index.duplicated(keep='first')]  # Drop duplicates

    # Align the index of solar_pos with X
    solar_pos.index = X.index


    #print(solar_pos)
    #for col in solar_pos.columns:
    #    print(col)
    #    if col == "elevation" or col == "azimuth":
    #        for index, value in solar_pos[col].items():
        #        print(index, value)
    #            if not pd.isna(value):
    #                print(f"Index {index} in column {col} is NOT NULL")


    X["solar_elevation"] = solar_pos["elevation"]
    #for index, value in X["solar_elevation"].items():
    #    if not pd.isna(value):
    #        print(f"Index {index} in column solar_elevation is NOT NULL")
    X["solar_azimuth"] = solar_pos["azimuth"]

    # row 7154 is NaN for solar_elevation and solar_azimuth
    # I actually don't know why, just fillna with the mean
    #X["solar_elevation"].fillna(X["solar_elevation"].mean(), inplace=True)
    X.fillna({"solar_elevation": X["solar_elevation"].mean(), "solar_azimuth": X["solar_azimuth"].mean()}, inplace=True)
    #X["solar_azimuth"].fillna(X["solar_azimuth"].mean(), inplace=True)

    return X

def create_interaction_features(X):
    X = X.copy()
    # Creating combined features
    X["temp_irradiation"] = X["T2m"] * X["G(i)"]
    X["wind_irradiation"] = X["WS10m"] * X["G(i)"]
    return X

def create_fourier_features(X):
    X = X.copy()
    # Might help in prediction if multiple years of data are used
    X["fourier1"] = np.sin(2 * np.pi * X["day_of_year"] / 365)
    X["fourier2"] = np.cos(2 * np.pi * X["day_of_year"] / 365)
    return X
