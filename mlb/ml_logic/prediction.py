import pandas as pd
from mlb.ml_logic.preprocessing import preprocessing_for_predictions

def build_X_new_preproc(pitcher_full_name, hitter_full_name):

    pitchers = pd.read_csv("../interface/data/pitchers.csv", index_col=0)
    hitters = pd.read_csv("../interface/data/hitters.csv", index_col=0)

    X_new = pd.concat([hitters[hitters.full_name == hitter_full_name].reset_index(),
            pitchers[pitchers.full_name == pitcher_full_name].reset_index()], axis=1)

    # Calculate handed_matchup
    X_new["handed_matchup"] = X_new.apply((lambda row: 0 if row["hitter_hand"] == row["pitcher_hand"] else 1), axis=1)

    # Calculate match_up_ab_count_delta
    X_new["match_up_ab_count_delta"] = X_new["pitcher_ab_count"] - X_new["hitter_ab_count"]

    # Remove columns
    X_new = X_new.drop(columns=["id", "full_name", "team_nickname", "primary_position",
                                "hitter_hand", "pitcher_hand", "pitcher_ab_count", "hitter_ab_count"])

    # Preprocess X_new
    X_train = pd.read_csv("../interface/data/X_train.csv", index_col=0)
    X_new_preproc = preprocessing_for_predictions(X_train, X_new)

    return X_new_preproc
