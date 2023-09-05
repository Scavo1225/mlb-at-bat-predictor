# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.impute import SimpleImputer


def create_preprocessor_pipeline() -> make_pipeline:
    """
    Take X_train and X_Test data with new features as input and return X_processed for train and test
    """
    ## Build pipeline for imputing and scaling

    # Num features
    num_col = ["hitter_previous_stats_szn_slug","rolling_1ab_slug", "rolling_3ab_slug",
            "rolling_10ab_slug", "pitcher_previous_stats_szn_bases", "rolling_1pitch_bases",
            "rolling_3pitch_bases", "rolling_10pitch_bases", "match_up_ab_count_delta",
            "hitter_success_speed", "pitcher_speed"]

    num_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        MinMaxScaler()
    )

    # Cat features
    cat_col = ["hitter_position"]
    cat_transformer = make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(sparse_output=True, drop="if_binary")
    )

    # Pipeline
    preproc_transformer = make_column_transformer(
        (num_transformer, num_col),
        (cat_transformer, cat_col),
        remainder='passthrough'
    )

    return preproc_transformer


def preprocessing_for_training(X_train, X_test):

    #Instanitate preprocessor
    preproc_transformer = create_preprocessor_pipeline()


    ## Fit on X_train and transform X_train and X_test
    X_train_preproc = preproc_transformer.fit_transform(X_train)
    X_test_preproc = preproc_transformer.transform(X_test)


    return X_train_preproc, X_test_preproc


def preprocessing_for_predictions(X_train, X_pred):
    
    preproc_transformer = create_preprocessor_pipeline()


    ##Fit on X_train and transform X_train and X_test
    preproc_transformer = preproc_transformer.fit(X_train)
    X_pred_preproc = preproc_transformer.transform(X_pred)


    return X_pred_preproc
