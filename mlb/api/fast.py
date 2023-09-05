import pandas as pd
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from mlb.ml_logic.registry import load_model
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
        pitcher_name : str,
        hitter_name: str,
    ):
    '''
    Make a prediction with probability calculation for a single at bat (hitter vs. pitcher)
    '''
    X_proc = build_X_new_preproc(pitcher_name, hitter_name)

    y_pred = app.state.model.predict(X_proc)
    y_proba = app.state.model.predict_proba(X_proc)[1]


    return {'prediction': y_pred,
            "probability": y_proba}


@app.get("/")
def root():
    return {
    'greeting': 'mlb_api_tests'
}
