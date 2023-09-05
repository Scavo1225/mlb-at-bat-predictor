# Libraries
import pandas as pd
import numpy as np
from colorama import Fore, Style
from typing import Tuple

from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score


def initialize_model(model_type='hgbc'):

    """
    Select type of model to intiailize a model
    'hgbc' = HistGradientBoostingClassifier,
    'xbg' = XGBoost Classifier
    'cb' = CatBoost model
    """
    if model_type not in ['hgbc','xbg','cb']:
        return """
                Please select one of the following: /n
                'hgbc' for HistGradientBoostingClassifier /n
                'xbg' = XGBoost Classifier /n
                'cb' = CatBoost model """

    if model_type == 'hgbc':
        model = HistGradientBoostingClassifier(loss="log_loss",
                                        learning_rate=0.1,
                                        max_iter=100,
                                        max_depth=None,
                                        l2_regularization=0,
                                        scoring="accuracy",
                                        validation_fraction=0.2,
                                        verbose=False,
                                        early_stopping=True,
                                        n_iter_no_change=10
                                        )

    return model


def train_model(model, X, y):
    model.fit(X, y)

    return model

def evaluate_model(model, X, y):
    y_pred = model.predict(X)

    print("✅ Test set evaluated, scoring on test set is as follows: \n")

    print("Accuracy =", round(accuracy_score(y, y_pred),2))
    print("Recall =", round(recall_score(y, y_pred),2))
    print("Precision =", round(precision_score(y, y_pred),2))

    return None
