import pandas as pd
import pickle


def data_preprocessing():
    # use historical radiation data + generated energy data to later on train the model

    # Load historical data: X
    with open("C:/Users/BRudo/solar_energy_forecast/data/raw data/historical_solar_data.pkl", "rb") as file:
        X = pickle.load(file)
    #print("X\n", X)

    # Load historical generated energy data: y
    y = pd.read_csv("C:/Users/BRudo/solar_energy_forecast/data/raw data/Actual_generation_2020.csv", delimiter=";", dtype=str)
    #print("y\n", y)


    y = y[["Start date", "Photovoltaics [MWh] Calculated resolutions"]]
    y = y.iloc[:-24,:] # remove last 24 rows (1 day), in order to have the full year 2020
    # 8784 rows = 8760h + 24h (because 2020 was a leap year)
    y = y.rename(columns={"Start date": "time", "Photovoltaics [MWh] Calculated resolutions": "energy_generated"})
    y["time"] = pd.to_datetime(y["time"], format="%b %d, %Y %I:%M %p")
    y["energy_generated"] = pd.to_numeric(y["energy_generated"].str.replace(",", ""), errors="coerce")
    #print("Y\n", y)

    # shifting the index (time) of X by 10 minutes to match the index of y
    X["time"] = pd.to_datetime(X["time"], format="%Y%m%d:%H%M")
    X["time"] = X["time"] - pd.Timedelta(minutes=10)
    #print("X\n", X)

    # take care of the big numbers XXXXX.XX that come from the wrongly interpreted dates in the energy_generated column from the original csv
    # Obviously this is not clean, it could have been other numbers that got wrongly interpreted, but hey, it's a first step
    # TOOK CARE OF with dtype=str in read_csv

    # Obviously just shifting the data by 10 minutes is not the best way to do it, but it's a first step


    #print(X.dtypes)
    #print("X describe:\n", X.describe())

    #print(y.dtypes)
    #print(max(y["energy_generated"]))
    #print("Y describe:\n", y.describe())

    return X, y
