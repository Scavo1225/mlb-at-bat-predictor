import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
# from mlb.ml_logic.registry import load_model

app = FastAPI()
# app.state.model = load_model()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)
###########################################
#API call functions
###########################################

@app.get("/")
def root():
    return {
    'greeting': 'mlb_api_tests'
}

# @app.get("/predict")
# def predict(
#         pickup_datetime: str,  # 2013-07-06 17:18:00
#         pickup_longitude: float,    # -73.950655
#         pickup_latitude: float,     # 40.783282
#         dropoff_longitude: float,   # -73.984365
#         dropoff_latitude: float,    # 40.769802
#         passenger_count: int
#     ):      # 1
#     """
#     Make a single course prediction.
#     Assumes `pickup_datetime` is provided as a string by the user in "%Y-%m-%d %H:%M:%S" format
#     Assumes `pickup_datetime` implicitly refers to the "US/Eastern" timezone (as any user in New York City would naturally write)
#     """
#     # YOUR CODE HERE
#     # Convert pickup_datetime to a pandas Timestamp in a specific timezone
#     pickup_datetime_timestamp = pd.Timestamp(pickup_datetime, tz='US/Eastern')
#    # Create a dictionary with user inputs
#     data = {
#         "pickup_datetime": [pickup_datetime_timestamp],
#         "pickup_longitude": [float(pickup_longitude)],
#         "pickup_latitude": [float(pickup_latitude)],
#         "dropoff_longitude": [float(dropoff_longitude)],
#         "dropoff_latitude": [float(dropoff_latitude)],
#         "passenger_count": [int(passenger_count)]
#     }

#     df = pd.DataFrame(data)

#     # Preprocess features
#     X_preprocessed = preprocess_features(df)

#     model = app.state.model

#     amount = model.predict(X_preprocessed)

#     return {'fare_amount': round(float(amount), 2)}
