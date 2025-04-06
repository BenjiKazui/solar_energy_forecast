import requests
import pickle
import json
import pandas as pd
import numpy as np
from datetime import date, datetime, UTC
from io import StringIO


def pull_historical_weather_data(save=False, save_path=None, start_year=None, end_year=None):
    """
    Pulls historical weather data. Years of data range from 2005 to 2020.
    Latitude and Longitude of Freiburg im Breisgau is being used.
    Documentation: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en
    """

    # Latitude and Longitude of Freiburg im Breisgau
    lat = "47.99"
    lon = "7.84"
    # API URL
    url = f"https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat={lat}&lon={lon}&startyear={str(start_year)}&endyear={str(end_year)}&outputformat=csv"

    # Fetch data
    response = requests.get(url)
    print("response code: ", response.status_code)

    # Check if the response is valid
    if response.status_code == 200:
        try:
            # Split the response into lines
            lines = response.text.split("\n")

            # Goal here: Get the column names and the actual data
            column_names = []
            data_lines = []

            # Loop over lines, if a line starts with "time" use the entries of that line for the column names
            for line in lines:
                if line.startswith("time"):  # Header row found
                    column_names = line.strip().split(",")
                    continue  # Skip adding the header to data_lines
                if column_names and not any(c.isalpha() for c in line.split(",")[:]):  
                    # Only keep rows that don’t contain letters
                    data_lines.append(line)

            if column_names is None:
                raise ValueError("Header not found in response data.")
        except Exception as e:
            print("Error while parsing CSV:", str(e))
    else:
        print(f"Error {response.status_code}: {response.text}")

    print("Extracted column_names: ", column_names)

    # get data ready to be moved into a df by joining the data
    csv_data = "\n".join(data_lines)
    df = pd.read_csv(StringIO(csv_data), names=column_names, header=None)

    # Remove footer metadata: Keep only rows where "time" is a valid timestamp
    df = df[df["time"].str.match(r"^\d{8}:\d{4}$", na=False)]

    # use datetime format for time column
    df["time"] = pd.to_datetime(df["time"], errors="coerce", format="%Y%m%d:%H%M")
    # use float64 format for G(i) column
    df["G(i)"] = pd.to_numeric(df["G(i)"], errors="coerce")

    # Get some more information on the data
    #print("DFDFDF", df)
    #print("dtypes\n", df.dtypes)
    #print("df.describe", df.describe())
    #print("NaNs\n", df.isna().sum())

    # drop unwanted columns
    df.drop(columns=["H_sun", "Int"], inplace=True)

    if save == True:
        # Save dataframe as pickle
        with open(save_path, "wb") as file:
            pickle.dump(df, file)

    return df


def date_to_milliseconds(date_string):
    """
    Helper function to convert custom date (YYYY-MM-DD HH:MM:SS) to Unix timestamp in milliseconds
    """
    #from datetime import datetime
    dt = datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S")
    return int(dt.timestamp() * 1000)


def get_available_timestamps(filter, region, resolution):
    """
    Fetches ALL available timestamps, which we will filter and then use to fetch the actual data
    """
    url = f"https://www.smard.de/app/chart_Data/{filter}/{region}/index_{resolution}.json"

    response = requests.get(url)

    # Check for status code and get a list of the timestamps if status code is fine
    if response.status_code == 200:
        timestamps = response.json().get("timestamps", [])
        return timestamps
    else:
        print(f"Failed to retrieve timestamps: {response.status_code}")
        return None


def pull_historical_energy_data(save=False, save_path=None, start_year=None, end_year=None):
    """
    In the SMARD API each timestamp corresponds to the start of an available dataset.
    Thus a single timestamp does not fetch a full time range — we need to collect data for multiple timestamps to cover a period.

    So, to get the energy data from this API we first need to get certain timestamps through a request.
    Once we have those timestamps we can use them in another request to get the actual data behind those timestamps.
    Using the region 'TransnetBW', which spans over the state of Baden-Württemberg - the state where Freiburg im Breisgau is located.
    Energy in MWh.
    Documentation: https://github.com/bundesAPI/smard-api
    """

    # Using a slightly expanded interval of time to look for timestamps that include the data we want.
    # Expanding by using roughly 1 past week before start_year and roughly 1 week after end_year to make
    # sure to get the entire interval of time we are interested in
    start_date = f"{start_year - 1}-12-24 00:00:00"
    end_date = f"{end_year + 1}-01-08 00:00:00"

    # API call: parameters
    filter = "4068" # code for energy generated by photovoltaics
    filterCopy = filter # must be specified according to the documentation
    region = "TransnetBW" # TSO for Baden-Württemberg (state) in Germany
    regionCopy = region # must be specified according to the documentation
    resolution = "hour" # hourly resolution of the data
    
    # Convert input dates to timestamps, because we need timestamps in milliseconds for the API to get all available timestamps
    start_ts = date_to_milliseconds(start_date)
    end_ts = date_to_milliseconds(end_date)
    print(f"Fetching data from {start_date} ({start_ts}) to {end_date} ({end_ts})")
    print(f"Which corresponds to {(end_ts - start_ts)/(3600*1000)} hours.")

    # Get all available timestamps
    available_timestamps = get_available_timestamps(filter=filter, region=region, resolution=resolution)
    if not available_timestamps:
        print("Problem with retrieving available timestamps.")
        return None
    
    # Filter all the pulled available timestamps to get only those matching our time interval of interest
    selected_timestamps = [ts for ts in available_timestamps if start_ts <= ts <= end_ts]
    if not selected_timestamps:
        print("No available timestamps in the given range.")

    print(f"Fetching data for {len(selected_timestamps)} timestamps.")

    # Finally fetch the actual data using the pulled and filtered timestamps
    all_data = []
    for timestamp in selected_timestamps:
        url = f"https://www.smard.de/app/chart_data/{filter}/{region}/{filterCopy}_{regionCopy}_{resolution}_{timestamp}.json"
        response = requests.get(url)

        if response.status_code == 200:
            data = response.json()
            all_data.append(data)
        else:
            print(f"Failed to retrieve data for timestamp {timestamp}")

    # Extract the time-series data
    data_list = []
    for entry in all_data:
        metadata = entry.get("smeta_data", {})
        series = entry.get("series", [])

        for timestamp, value in series:
            utc_date = datetime.fromtimestamp(timestamp / 1000, UTC).strftime("%Y-%m-%d %H:%M:%S")
            data_list.append({
                "timestamp": timestamp,
                "datetime": utc_date,
                "value": value,
                "version": metadata.get("version", ""),
                "created": metadata.get("created", "")
            })

    df = pd.DataFrame(data_list)

    first_timestamp = df["datetime"].iloc[0]
    last_timestamp = df["datetime"].iloc[-1]

    print(f"Given the start_date {start_date} and end_date {end_date} the available timestamps are ranging from {first_timestamp} to {last_timestamp}")

    # The returned df (and all_data) includes data for the time interval of interest, but it also includes data outside of that time interval of interest
    # Matching the data properly is done in the next step in the pipeline with functions from data_preprocessing.py
    return df, all_data


def pull_future_weather_data(save=False, save_path=None):
    """
    Potentially we could also fetch a weather forecast and do predictions on that forecast.
    Pros: It's cool to try
    Cons: It's really hard to interpret and get good results i think, because you are predicting (energy) on another prediction (weather).
          Weather Prediction could be bad, then so is the energy prediction.
          Also, for evaluating the prediction, we would need to wait for time to pass in order to see the actual energy generation for that time period.
          There is only roughly 14 days of weather predictions out there, and then those are actually not too accurate (I THINK), might need to look into it further

    Documentation: https://open-meteo.com/en/docs      
    """

    # Latitude/Longitude for Freiburg im Breisgau
    lat = "47.99"
    lon = "7.84"
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&hourly=shortwave_radiation&hourly=temperature_2m&hourly=wind_speed_10m"

    response = requests.get(url)

    data = json.loads(response.text)

    if response.status_code == 200:
        try:
            lat = data["latitude"]
            lon = data["longitude"]
            time = data["hourly"]["time"]
            temperature_2m = data["hourly"]["temperature_2m"]
            wind_speed_10m = data["hourly"]["wind_speed_10m"]
            shortwave_radiation = data["hourly"]["shortwave_radiation"]
        except Exception as e:
            print("Error while parsing JSON:", str(e))
    else:
        print(f"Error {response.status_code}: {response.text}")

    print(np.mean(shortwave_radiation))

    # convert unit of windspeed from km/h to m/s
    wind_speed_10m = [round(wind_speed / 3.6, 2) for wind_speed in wind_speed_10m]

    df = pd.DataFrame.from_dict({"time": time, "shortwave_radiation": shortwave_radiation, "temperature_2m": temperature_2m, "wind_speed_10m": wind_speed_10m})

    if save == True and save_path is not None:
        with open(save_path, "wb") as file:
            pickle.dump(df, file)

    return df


# Functions to load pickle data from local machine
def load_local_data(local_path):
    with open(local_path, "rb") as file:
        df = pickle.load(file)
    return df