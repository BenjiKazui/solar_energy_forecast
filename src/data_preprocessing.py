import pandas as pd


def data_preprocessing(hist_weather_data=None, hist_energy_data=None, future_weather_data=None):
    """
    Shifts the time column of the weather data back by 10 minutes. From 00:10 to 00:00 for example. In order to have both timestamps use 0 for the minute-timestamp.
    Also filters the energy data, which includes more timestamps/data than the weather data, to use only the timestamps used in the weather data.
    Converts time columns to datetime format, drops unwanted columns and renames columns.
    """

    # The jrc-API (weather data) gives hourly data with timestamps like these: HOUR:MINUTE -> 00:10, 01:10, .. 23:10
    # The Smard-API (energy data) gives hourly data with timestamps like these: HOUR:MINUTE -> 00:00, 01:00, .. 23:00
    # Solution: Shifting the values of column time of the weather data by 10 minutes to match the timestamps of the energy data
    # Obviously just shifting the data by 10 minutes is not the best way to do it, but it's a first solution
    hist_weather_data["time"] = pd.to_datetime(hist_weather_data["time"], format="%Y%m%d:%H%M")
    hist_weather_data["time"] = hist_weather_data["time"] - pd.Timedelta(minutes=10)

    # Drop unwanted columns and rename columns
    hist_energy_data = hist_energy_data.drop(columns=["timestamp", "version", "created"])
    hist_energy_data = hist_energy_data.rename(columns={"datetime": "time", "value": "energy"})
    hist_energy_data["time"] = pd.to_datetime(hist_energy_data["time"])

    # We can't just get the data for the timestamps we actually want to use from the smard-API, because of how the smard-API is structured
    # That is why we filter the energy data to use only data for the same timestamps we use for the weather data
    hist_energy_data = hist_energy_data[hist_energy_data["time"].isin(hist_weather_data["time"])]
    hist_energy_data = hist_energy_data.reset_index(drop=True)

    return hist_weather_data, hist_energy_data