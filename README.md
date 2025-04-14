☀️ **Solar Energy Forecast – Baden-Württemberg, Germany**  
This project aims to predict solar energy production in Baden-Württemberg (state of Germany) using weather data from Freiburg im Breisgau (city in Germany). Right now the test data consists only of historical data and is not yet based on weather forecasts. Using actual weather forecasts to predict the solar energy generation on those forecasts is the next step. As of right now this project is more something I play around once in a while and some proof of work.

🗺️ Region: Baden-Württemberg, Germany

🌤️ Input Data: Historical weather data from Freiburg im Breisgau

⚡ Target Variable: Actual solar energy production of Baden-Württemberg

🤖 Baseline-Model: Linear Regression  
🤖 Model: XGBoost Regressor

🧪 Validation: Cross-validation used for hyperparameter tuning

📈 Goal: Predict solar energy generation using weather data

Although the weather data is only taken from a single city (Freiburg), it serves as a representative proxy for the entire state in this prototype.

📊 Data Sources  
Weather data: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en

Solar energy data: https://github.com/bundesAPI/smard-api

⚙️ **Workflow**  
Data ingestion and preprocessing

Feature engineering (radiation, temperature, wind speed, time-based, sun position, lag and rolling mean)

Hyperparameter tuning via cross-validation

Prediction and evaluation using unseen test data

🚧 Limitations  
Weather input limited to Freiburg — not representative of all local variations across the state

🧠 Future Improvements
Use weather forecasts to predict solar energy generation on

Try alternative models (e.g. neural networks)

Deploy as an interactive dashboard or API

Perform deeper error analysis and model monitoring

Possibly integrate more weather stations across the state
or any other idea to solve the conflict that we use weather data
from one city to predict solar energy generation of an entire state
(Could look for other, more fitting data too)
