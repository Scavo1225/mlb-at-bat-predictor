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
from mlb.ml_logic.data import create_dataset, build_data_to_predict
from mlb.ml_logic.features import create_features
from mlb.ml_logic.preprocessing import preprocessing_for_training
from mlb.ml_logic.modeling import initialize_model, train_model, evaluate_model
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

    print(Fore.MAGENTA + "\n ⭐️ Prediction tables created...." + Style.RESET_ALL)

    print(Fore.MAGENTA + "\n ⭐️ Dataset retrieved and features created...." + Style.RESET_ALL)

    X = df.drop(columns=["y_target"])
    y = df.y_target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size)

    X_train.to_csv('mlb/interface/data/X_train.csv')

    build_data_to_predict()

    print(Fore.MAGENTA + "\n ⭐️ Training, test sets, and prediction tables created...." + Style.RESET_ALL)

    X_train_processed, X_test_processed = preprocessing_for_training(X_train, X_test)

    print(Fore.MAGENTA + "\n ⭐️ Feature and targets created and preprocessed" + Style.RESET_ALL)

    print(Fore.MAGENTA + f"\n ⭐️ Initializing model trained with {X_train_processed.shape[0]} at bats..." + Style.RESET_ALL)

    model = load_model()

    if model is None:
        model = initialize_model(model_type=model_type)
        model = train_model(model, X_train_processed, y_train)

    print(Fore.MAGENTA + f"\n ⭐️ Validating model with {X_train_processed.shape[0]} at bats..." + Style.RESET_ALL)

    cv_results = cross_validate(model, X_train_processed, y_train, cv=5, n_jobs=-1, scoring=["accuracy", "recall", "precision"])

    print("✅ model loaded with following metrics \n")

    print("Accuracy =", round(cv_results["test_accuracy"].mean(),2))
    print("Recall =", round(cv_results["test_recall"].mean(),2))
    print("Precision =", round(cv_results["test_precision"].mean(),2))

    # Save model weights on the hard drive and mlflow
    save_model(model=model)

    print("✅ Evaluating testing data set...... \n")

    scoring =  evaluate_model(model, X_test_processed, y_test)

    return None


if __name__ == '__main__':
    get_trained_model(model_type='hgbc', test_size=0.30)
