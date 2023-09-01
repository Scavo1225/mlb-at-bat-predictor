import numpy as np
import pandas as pd

from pathlib import Path
from colorama import Fore, Style
from dateutil.parser import parse

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

from mlb.params import *
from mlb.ml_logic.data import create_dataset
from mlb.ml_logic.features import create_features
from mlb.ml_logic.preprocessing import preprocessing_for_training,  preprocessing_for_prediction
from mlb.ml_logic.modeling import initialize_model, train_model
from mlb.ml_logic.registry import load_model, save_model, save_results
from mlb.ml_logic.registry import mlflow_run, mlflow_transition_model

def get_preprocessed_data(test_size=0.30):
    '''
    Parse through JSON files to pull raw data
    clean and merge tabular data
    drop unneeded and/or redundant features
    create modeling features
    return preprocessed X_train, X_test, y_train, y_test

    '''

    print(Fore.MAGENTA + "\n ⭐️ Pulling dataset and preprocessing...." + Style.RESET_ALL)

    raw_df = create_dataset()
    df = create_features(raw_df)

    X = df.drop(columns=["y_target"])
    y = df.y_target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train_processed, X_test_processed = preprocessing_for_training(X_train, X_test)

    return X_train_processed, X_test_processed, y_train, y_test

@mlflow_run
def get_trained_model(model_type='hgbc', test_size=0.30):

    X_train_processed, X_test_processed, y_train, y_test = get_preprocessed_data(test_size=test_size)

    print(Fore.MAGENTA + "\n ⭐️ Training model with {x_train_process.shape[0]} at bats..." + Style.RESET_ALL)

    model = initialize_model(model_type=model_type)
    model = train_model(model, X_train_processed, y_train)

    return model


# def get_model_evaluation():

#     '''
#     Load model from mlflow and evaluate the current production model
#     '''


if __name__ == '__main__':
    get_preprocessed_data(test_size=0.30)
    get_trained_model(model_type='hgbc', test_size=0.30)
