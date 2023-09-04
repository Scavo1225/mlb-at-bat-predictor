import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mlb.ml_logic.registry import load_model
from mlb.ml_logic.features import create_features
from mlb.ml_logic.preprocessing import preprocessing_for_predictions

app = FastAPI()
app.state.model = load_model()

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

@app.get("/predict")
def predict(
        pitcher_name : str,
        hitter_name: str,
    ):

    data = {
        "pitcher_name": [pitcher_name],
        "hitter_name": [hitter_name],
    }

    df = pd.DataFrame(data)


    #Preprocess features
    #X_preprocessed = preprocess_features(df)
    model = app.state.model
    hit_prediction = model.predict(X_preprocessed)
    return {'Base': hit_prediction }
