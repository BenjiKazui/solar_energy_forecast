import requests
import json
import pandas as pd
import pickle
import numpy as np

# API URL
url = "https://api.open-meteo.com/v1/forecast?latitude=47.99&longitude=7.84&hourly=shortwave_radiation&hourly=temperature_2m&hourly=wind_speed_10m"

response = requests.get(url)
print(response.status_code)

data = json.loads(response.text)

if response.status_code == 200:
    try:
        lat = data["latitude"]
        lon = data["longitude"]
        timezone = data["timezone"]
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
#print("Generated df:\n", df)

with open("C:/Users/BRudo/solar_energy_forecast/data/raw data/forecasted_solar_data.pkl", "wb") as file:
    pickle.dump(df, file)