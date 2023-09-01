import pandas as pd

def create_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    DataFrame containing all features used to pass to the models
    Creating rolling hitting and pitching statistics leading into future at bat as features for prediction
    '''

    #Creating hitter stats up to at bat
    df['hitter_previous_stats_szn'] = df.groupby("hitter_id")["y_target"].cumsum() / (df.groupby('hitter_id').cumcount() + 1)
    df['rolling_1ab'] = df.groupby("hitter_id")["y_target"].transform(lambda x: x.shift().rolling(1, min_periods=1).mean())
    df['rolling_3ab'] = df.groupby("hitter_id")["y_target"].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    df['rolling_10ab'] = df.groupby("hitter_id")["y_target"].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())

    df['hitter_previous_stats_szn_slug'] = df.groupby("hitter_id")["mc_target"].cumsum() / (df.groupby('hitter_id').cumcount() + 1)
    df['rolling_1ab_slug'] = df.groupby("hitter_id")["mc_target"].transform(lambda x: x.shift().rolling(1, min_periods=1).mean())
    df['rolling_3ab_slug'] = df.groupby("hitter_id")["mc_target"].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    df['rolling_10ab_slug'] = df.groupby("hitter_id")["mc_target"].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())

    # Calculate stats of pitcher before he pitchs
    df['pitcher_previous_stats_szn'] = df.groupby("pitcher_id")["y_target"].cumsum() / (df.groupby('pitcher_id').cumcount() + 1)
    df['rolling_1pitch'] = df.groupby("pitcher_id")["y_target"].transform(lambda x: x.shift().rolling(1, min_periods=1).mean())
    df['rolling_3pitch'] = df.groupby("pitcher_id")["y_target"].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    df['rolling_10pitch'] = df.groupby("pitcher_id")["y_target"].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())

    df['pitcher_previous_stats_szn_bases'] = df.groupby("pitcher_id")["mc_target"].cumsum() / (df.groupby('pitcher_id').cumcount() + 1)
    df['rolling_1pitch_bases'] = df.groupby("pitcher_id")["mc_target"].transform(lambda x: x.shift().rolling(1, min_periods=1).mean())
    df['rolling_3pitch_bases'] = df.groupby("pitcher_id")["mc_target"].transform(lambda x: x.shift().rolling(3, min_periods=1).mean())
    df['rolling_10pitch_bases'] = df.groupby("pitcher_id")["mc_target"].transform(lambda x: x.shift().rolling(10, min_periods=1).mean())


    #Create handed matchups
    df["handed_matchup"] = df["hitter_hand"] + df["pitcher_hand"]
    df["handed_matchup"] = df.handed_matchup.apply((lambda x: 0 if x[0] == x[1] else 1))
    df = df.drop(columns=["hitter_hand", "pitcher_hand"])

    #total hitter and pitcher at bats prior to at bat
    df["hitter_ab_count"] = df.groupby('hitter_id').cumcount()
    df["pitcher_ab_count"] = df.groupby('pitcher_id').cumcount()
    df["match_up_ab_count_delta"] = df["pitcher_ab_count"] - df["hitter_ab_count"]

    #Balls and strikes efficiencies, batters
    df["reverse_pitch_class"] = df["pitch_class"].apply(lambda x: 0 if x == 1 else 1)

    df["hitter_strikes_atb"] = (df.groupby("hitter_id")["reverse_pitch_class"].cumsum())
    df["hitter_balls_atb"] = (df.groupby("hitter_id")["pitch_class"].cumsum())

    df["hitter_strikes_only_results"] = df["y_target"] * df["reverse_pitch_class"]
    df["hitter_balls_only_results"] = df["y_target"] * df["pitch_class"]

    df["hitter_strikes_results"] = (df.groupby("hitter_id")["hitter_strikes_only_results"].cumsum())
    df["hitter_balls_results"] = (df.groupby("hitter_id")["hitter_balls_only_results"].cumsum())

    df["hitter_strikes_eff"] = df["hitter_strikes_results"] / df["hitter_strikes_atb"]
    df["hitter_balls_eff"] = df["hitter_balls_results"] / df["hitter_balls_atb"]

    #Balls and strikes spread pitchers

    df["pitcher_strikes_spread"] = (df.groupby("pitcher_id")["reverse_pitch_class"].cumsum()) / (df.groupby("pitcher_id")["pitcher_id"].cumcount() + 1)
    df["pitcher_balls_spread"] = (df.groupby("pitcher_id")["pitch_class"].cumsum()) / (df.groupby("pitcher_id")["pitcher_id"].cumcount() + 1)


    #pitch speed stats
    df["hitter_speeds_class1"] = df["pitch_speed_mph"] * df["y_target"]
    df["hitter_success_speed"] = (df.groupby("hitter_id")["hitter_speeds_class1"].cumsum()) / (df.groupby("hitter_id")["y_target"].cumsum())

    df["pitcher_speed"] = (df.groupby("pitcher_id")["pitch_speed_mph"].cumsum())/ (df.groupby("pitcher_id").cumcount())

    #Pitch type stats
    df["reverse_pitch_type_cat"] = df["pitch_type_cat"].apply(lambda x: 0 if x == 1 else 1)

    df["hitter_fast_atb"] = (df.groupby("hitter_id")["reverse_pitch_type_cat"].cumsum())
    df["hitter_offspeed_atb"] = (df.groupby("hitter_id")["pitch_type_cat"].cumsum())

    df["hitter_fast_only_results"] = df["y_target"] * df["reverse_pitch_type_cat"]
    df["hitter_offspeed_only_results"] = df["y_target"] * df["pitch_type_cat"]

    df["hitter_fast_results"] = (df.groupby("hitter_id")["hitter_fast_only_results"].cumsum())
    df["hitter_offspeed_results"] = (df.groupby("hitter_id")["hitter_offspeed_only_results"].cumsum())

    df["hitter_fast_eff"] = df["hitter_fast_results"] / df["hitter_fast_atb"]
    df["hitter_offspeed_eff"] = df["hitter_offspeed_results"] / df["hitter_offspeed_atb"]

    #Balls and strikes spread pitchers

    df["pitcher_fast_spread"] = (df.groupby("pitcher_id")["reverse_pitch_type_cat"].cumsum()) / (df.groupby("pitcher_id")["pitcher_id"].cumcount() + 1)
    df["pitcher_offspeed_spread"] = (df.groupby("pitcher_id")["pitch_type_cat"].cumsum()) / (df.groupby("pitcher_id")["pitcher_id"].cumcount() + 1)

    #dropping redundant columns to avoid multi-colinearity

    df = df.drop(columns=["pitch_class", "reverse_pitch_class", "hitter_strikes_atb", "hitter_balls_atb", "hitter_strikes_only_results",
                            "hitter_balls_only_results", "hitter_strikes_results", "hitter_balls_results", "hitter_speeds_class1",
                            "reverse_pitch_type_cat", "hitter_fast_atb", "hitter_offspeed_atb", "hitter_fast_only_results","hitter_offspeed_only_results","hitter_fast_results", "hitter_offspeed_results"])

    final_modeling_columns = ['temp_f', 'humidity',
                    'pitcher_pitch_count_at_bat_start', 'outs_at_start',
                    'hitter_position', 'hitter_previous_stats_szn', 'rolling_1ab',
                    'rolling_3ab', 'rolling_10ab', 'hitter_previous_stats_szn_slug',
                    'rolling_1ab_slug', 'rolling_3ab_slug', 'rolling_10ab_slug',
                    'pitcher_previous_stats_szn', 'rolling_1pitch', 'rolling_3pitch',
                    'rolling_10pitch', 'pitcher_previous_stats_szn_bases',
                    'rolling_1pitch_bases', 'rolling_3pitch_bases', 'rolling_10pitch_bases',
                    'handed_matchup','match_up_ab_count_delta', 'hitter_strikes_eff', 'hitter_balls_eff',
                    'pitcher_strikes_spread', 'pitcher_balls_spread',
                    'hitter_success_speed', 'pitcher_speed', 'hitter_fast_eff',
                    'hitter_offspeed_eff', 'pitcher_fast_spread',
                    'pitcher_offspeed_spread', 'y_target']

    df = df[final_modeling_columns]

    return df
