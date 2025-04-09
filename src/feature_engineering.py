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


def create_approximate_capacity(X):
    """
    I haven't found a good source where one can export data regarding the total capacity of PV systems in Baden-Württemberg.
    But there are Visualizations available on the internet, this function right now uses a visualization of yearly installed capacities of PV systems in Baden-Württemberg and sums them up.
    Page 15 in: https://www.baden-wuerttemberg.de/fileadmin/redaktion/m-um/intern/Dateien/Dokumente/2_Presse_und_Service/Publikationen/Energie/Eneuerbare-Energien-2022.pdf
    Worth a try: Get into contact with the publishers and ask if I can have the actual data.
    For now we just say the total installed capacity only changes once a year, on the first of January.
    This is obviously not true, but it is a first approximation.
    """
    # Approximation: mapping of years and total installed capacity of PV systems in Baden-Württemberg (BW)
    BW_PV_capacity = {"2005": 197, "2006": 389, "2007": 650, "2008": 1044, "2009": 1672, "2010": 2706, "2011": 3631, "2012": 4218, "2013": 4565,
                      "2014": 4814, "2015": 4977, "2016": 5122, "2017": 5328, "2018": 5634, "2019": 6062, "2020": 6684, "2021": 7309, "2022": 8129}

    # assign the total installed capacity to the dataframe for each year
    # Note: This is a very rough approximation and should be replaced with actual data if available
    for year in X["time"].dt.year.unique():
        if str(year) in BW_PV_capacity.keys():
            X.loc[X["time"].dt.year == year, "PV_capacity"] = BW_PV_capacity[str(year)]
    return X


def create_features(df, feature_names):
    """
    Create features from provided dataframe and feature names that are passed in.
    """
    if "PV_capacity" in feature_names:
        df = create_approximate_capacity(df)
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
        raise ValueError("Unknown feature name, use 'PV_capacity', 'time_based', 'lag', 'rolling', 'sun_position', 'interaction' or 'fourier'")
    return df