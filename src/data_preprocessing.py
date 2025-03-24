import pandas as pd
import pickle
import pytz # for time zone handling

def data_preprocessing_2(hist_weather_data=None, hist_energy_data=None, future_weather_data=None):
    hist_weather_data["time"] = pd.to_datetime(hist_weather_data["time"], format="%Y%m%d:%H%M")

    # shifting the index (time) of X by 10 minutes to match the index of y
    # Obviously just shifting the data by 10 minutes is not the best way to do it, but it's a first step
    hist_weather_data["time"] = hist_weather_data["time"] - pd.Timedelta(minutes=10)

    print("hist_weather_data:\n", hist_weather_data)


    hist_energy_data = hist_energy_data.drop(columns=["timestamp", "version", "created"])
    hist_energy_data = hist_energy_data.rename(columns={"datetime": "time", "value": "energy"})
    hist_energy_data["time"] = pd.to_datetime(hist_energy_data["time"])

    #import matplotlib.pyplot as plt
    #fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    #axes[0].plot(hist_energy_data.index, hist_energy_data.energy)
    #axes[0].set_title("before")

    print("hist_energy_data:\n", hist_energy_data)

    hist_energy_data = hist_energy_data[hist_energy_data["time"].isin(hist_weather_data["time"])]
    hist_energy_data = hist_energy_data.reset_index(drop=True)

    #axes[1].plot(hist_energy_data.index, hist_energy_data.energy)
    #axes[1].set_title("after")

    #plt.tight_layout()
    #plt.show()

    print("hist_energy_data:\n", hist_energy_data)

    return hist_weather_data, hist_energy_data

    #return X, y


def data_preprocessing():

    # Load historical data: X
    with open("C:/Users/BRudo/solar_energy_forecast/data/raw data/historical_solar_data.pkl", "rb") as file:
        X = pickle.load(file)

    # Load historical generated energy data: y
    y = pd.read_csv("C:/Users/BRudo/solar_energy_forecast/data/raw data/Actual_generation_2020.csv", delimiter=";", dtype=str)

    y = y[["Start date", "Photovoltaics [MWh] Calculated resolutions"]]
    # remove last 24 rows (1 day), in order to have the full year 2020
    # 8784 rows = 8760h + 24h (because 2020 was a leap year)
    y = y.iloc[:-24,:]
    y = y.rename(columns={"Start date": "time", "Photovoltaics [MWh] Calculated resolutions": "energy_generated"})
    y["time"] = pd.to_datetime(y["time"], format="%b %d, %Y %I:%M %p")
    y["energy_generated"] = pd.to_numeric(y["energy_generated"].str.replace(",", ""), errors="coerce")

    # shifting the index (time) of X by 10 minutes to match the index of y
    # Obviously just shifting the data by 10 minutes is not the best way to do it, but it's a first step
    X["time"] = pd.to_datetime(X["time"], format="%Y%m%d:%H%M")
    X["time"] = X["time"] - pd.Timedelta(minutes=10)

    print("TIMEZONE")
    print(X["time"].dt.tz)

    if X["time"].dt.tz is None:
        X["time"] = X["time"].dt.tz_localize("Europe/Berlin", nonexistent="shift_forward", ambiguous=False)
        print("Timezone after localization:", X["time"].dt.tz)
        X["time"] = X["time"].dt.tz_convert("UTC")
        print("Timezone after conversion to UTC:", X["time"].dt.tz)

    print(X.index.duplicated().sum())

    return X, y
