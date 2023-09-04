import numpy as np
import pandas as pd

# from pathlib import Path
from colorama import Fore, Style
# from dateutil.parser import parse

from sklearn.model_selection import train_test_split, cross_validate
# from sklearn.compose import make_column_transformer
# from sklearn.pipeline import make_pipeline
# from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
# from sklearn.impute import SimpleImputer

from mlb.params import *
from mlb.ml_logic.data import create_dataset
from mlb.ml_logic.features import create_features
from mlb.ml_logic.preprocessing import preprocessing_for_training
from mlb.ml_logic.modeling import initialize_model, train_model
from mlb.ml_logic.registry import load_model, save_model
from mlb.ml_logic.registry import mlflow_run


@mlflow_run
def get_trained_model(model_type='hgbc', test_size=0.30):
    '''
    Parse through JSON files to pull raw data
    clean and merge tabular data
    drop unneeded and/or redundant features
    create modeling features
    create X_train, X_test, y_train, y_test
    load model from cloud; if no model available, create one.

    '''
    print(Fore.MAGENTA + "\n ⭐️ Pulling dataset and preprocessing...." + Style.RESET_ALL)

    raw_df = create_dataset()
    df = create_features(raw_df)

    print(Fore.MAGENTA + "\n ⭐️ Dataset retrieved and features created...." + Style.RESET_ALL)

    df.to_csv('preproc_data/test_data/df.csv')
    raw_df.to_csv('preproc_data/test_data/df_raw.csv')

    X = df.drop(columns=["y_target"])
    y = df.y_target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train_processed, X_test_processed = preprocessing_for_training(X_train, X_test)

    print(Fore.MAGENTA + "\n ⭐️ Feature and targets created and preprocessed" + Style.RESET_ALL)

    print(Fore.MAGENTA + f"\n ⭐️ Training model with {X_train_processed.shape[0]} at bats..." + Style.RESET_ALL)

    model = load_model()

    if model is None:
        model = initialize_model(model_type=model_type)
        model = train_model(model, X_train_processed, y_train)

    cv_results = cross_validate(model, X_train_processed, y_train, cv=5, n_jobs=-1, scoring=["accuracy", "recall", "precision"])

    print("✅ train() done with folowing metrics \n")

    print("Accuracy =", cv_results["test_accuracy"].mean())
    print("Recall =", cv_results["test_recall"].mean())
    print("Precision =", cv_results["test_precision"].mean())

    # Save model weights on the hard drive and mlflow
    save_model(model=model)

    return None

# def get_model_evaluation():

#     '''
#     Load model from mlflow and evaluate the current production model
#     '''


if __name__ == '__main__':
    get_trained_model(model_type='hgbc', test_size=0.30)
