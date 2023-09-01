# Libraries
import pandas as pd

from sklearn.ensemble import HistGradientBoostingClassifier

def hgbc_model(X_train_preproc: pd.DataFrame,
               y_train: pd.DataFrame
               ):

    """
    Take X_train_preproc and y_train, hgbc model parameter as input to return
    the fit model
    """

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

    ## Fiting model on X_train
    model.fit(X_train_preproc, y_train)

    return model
