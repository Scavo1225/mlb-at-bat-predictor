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

def preprocessing(data: pd.DataFrame) -> pd.DataFrame:
    """
    Take df data with new features as input and return X_train_preproc,
    y_train, X_test_preproc, y_test)
    """

    ## Split X and y
    X = data.drop(columns=["y_target"])
    y = data.y_target

    ## Slit train / test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30)

    ## Build pipeline for imputing and scaling

    # Num features
    num_col = ["temp_f", "humidity", "pitcher_pitch_count_at_bat_start", "outs_at_start",
               "hitter_previous_stats_szn_slug","rolling_1ab_slug", "rolling_3ab_slug",
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

    ## Fiting on X_train
    preproc_transformer.fit(X_train)

    ## Transform X_train and X_test + convert them as dataframe
    X_train_preproc = preproc_transformer.transform(X_train)
    X_train_preproc = pd.DataFrame(X_train_preproc,
                                   columns=preproc_transformer.get_feature_names_out(),
                                   index=X_train.index)

    X_test_preproc = preproc_transformer.transform(X_test)
    X_test_preproc = pd.DataFrame(X_test_preproc,
                                  columns=preproc_transformer.get_feature_names_out(),
                                  index=X_test.index)

    return X_train_preproc, X_test_preproc, y_train, y_test, preproc_transformer
