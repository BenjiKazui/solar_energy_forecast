import requests
import json
from io import StringIO
import pandas as pd
import pickle

# API URL
url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc?lat=47.99&lon=7.84&startyear=2020&endyear=2020&outputformat=csv"

# Fetch data
response = requests.get(url)

print("response code: ", response.status_code)

# Check if the response is valid
if response.status_code == 200:
    try:
        # Split the response into lines
        lines = response.text.split("\n")

        header_metadata = []
        data_lines = []
        header_found = False
        footer_metadata = []
        data_cleaned = []
        column_names = []

        for line in lines:
            if line.startswith("time"):
                header_found = True
                column_names = line.strip().split(",")
                data_lines.append(line)
                continue
            if not header_found:
                header_metadata.append(line)
            else:
                data_lines.append(line)

        for row in data_lines:
            if any(c.isalpha() for c in row) and not row.startswith("time"):
                footer_metadata.append(row)
            else:
                data_cleaned.append(row)

        print("Extracted column_names: ", column_names)

        csv_data = "\n".join(data_cleaned)
        df = pd.read_csv(StringIO(csv_data), names=column_names, header=0)

    except Exception as e:
        print("Error while parsing CSV:", str(e))
else:
    print(f"Error {response.status_code}: {response.text}")

df.drop(columns=["H_sun", "Int"], inplace=True)

# Save dataframe as pickle
with open("C:/Users/BRudo/solar_energy_forecast/data/raw data/historical_solar_data.pkl", "wb") as file:
    pickle.dump(df, file)
