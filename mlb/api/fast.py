import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mlb.ml_logic.registry import load_model
from mlb.ml_logic.features import create_features
from mlb.ml_logic.preprocessing import preprocessing_for_predictions
from mlb.ml_logic.prediction import build_X_new_preproc

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

@app.get("/predict")
def predict(
        pitcher_full_name : str,
        hitter_full_name: str,
    ):
    '''
    Make a prediction with probability calculation for a single at bat (hitter vs. pitcher)
    '''
    X_proc = build_X_new_preproc(pitcher_full_name, hitter_full_name)

    y_pred = app.state.model.predict(X_proc)
    y_proba = app.state.model.predict_proba(X_proc)[1]

    if y_pred == 1:
        prediction = "Batter wins this at bat!"
        proba = f"Batter is projected to win this at bat, with a {y_proba} probability"

    else:
        prediction = "Pitcher wins this at bat!"
        proba = f"Pitcher is expected to win the at bat; batter has a {y_proba} probability to be successful"


    return {'prediction': prediction,
            "probability": proba}


@app.get("/")
def root():
    return {
    'greeting': 'mlb_api_tests'
}
