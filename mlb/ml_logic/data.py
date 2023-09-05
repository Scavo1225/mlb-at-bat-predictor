import pandas as pd
import json
import numpy as np
import os


def raw_data_parse(filepath='raw_data/plate_app_data/') -> pd.DataFrame:

    '''
    Provide a directory to JSON files from API call source Sport Radar - Play by Play
    Parsing through JSON to pull the key data into a dataframe in raw form for all at bats in the game set

    '''
    directory = filepath
    data_list = []

    for filename in os.listdir(directory):
        file_ = os.path.join(directory, filename)
        with open(file_) as user_file:
            gdata = user_file.read()
        gdata = json.loads(gdata)

        for ing in range(1, len(gdata["game"]["innings"])):
            for hlf in range(2):
                for atb in range(len(gdata["game"]["innings"][ing]["halfs"][hlf]["events"])):

                    if 'at_bat' not in gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]:
                        continue

                    if 'pitcher_id' not in gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]:
                        continue

                    id = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["id"]
                    game_id = gdata["game"]["id"]

                    hitter_id = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["hitter_id"]
                    hitter_hand = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["hitter_hand"]

                    pitcher_id = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["pitcher_id"]
                    pitcher_hand = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["pitcher_hand"]

                    if 'description' not in gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]:
                        continue

                    description = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["description"]

                    try:
                        at_bat_end_time = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["wall_clock"]["end_time"]
                    except: at_bat_end_time = np.nan

                    try:
                        pitch_location_zone = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["mlb_pitch_data"]["zone"]
                    except:
                        pitch_location_zone = np.nan

                    try:
                        pitch_type_des = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["mlb_pitch_data"]["description"]
                    except:
                        pitch_type_des = np.nan

                    try:
                        pitch_speed_mph = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["pitcher"]["pitch_speed"]
                    except:
                        pitch_speed_mph = np.nan

                    try:
                        pitch_count_at_bat = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["count"]["pitch_count"]
                    except:
                        pitch_count_at_bat = np.nan

                    try:
                        pitcher_pitch_count_at_bat_start = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["pitcher"]["pitch_count"] - pitch_count_at_bat
                    except:
                        pitcher_pitch_count_at_bat_start = np.nan

                    try:
                        if atb == 0:
                            outs_at_start = 0
                        elif pitch_count_at_bat == 1:
                            outs_at_start = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb-1]["at_bat"]["events"][-1]["count"]["outs"]
                        else:
                            outs_at_start = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][0]["count"]["outs"]

                    except:
                        outs_at_start = np.nan


                    data = {'id': id,
                            'game_id': game_id,
                            'hitter_id': hitter_id,
                            'hitter_hand' :hitter_hand,
                            'pitcher_id': pitcher_id,
                            'pitcher_hand': pitcher_hand,
                            'description': description,
                            'at_bat_end_time': at_bat_end_time,
                            'pitch_location_zone': pitch_location_zone,
                            'pitch_speed_mph': pitch_speed_mph,
                            'pitch_count_at_bat': pitch_count_at_bat,
                            'pitcher_pitch_count_at_bat_start': pitcher_pitch_count_at_bat_start,
                            'outs_at_start': outs_at_start,
                            'pitch_type_des': pitch_type_des
                    }

                    data_list.append(data)

    df = pd.DataFrame(data_list)

    return df


def set_targets(df: pd.DataFrame) -> pd.DataFrame:
    '''Provide raw data DataFrame and return DataFrame with the outcome codes and classification targets
    Both binary and multi-class target

    outcome codes:
    walk - four balls on at bat, free base
    HBP - hit by pitch, free base
    1B - single
    2B - double
    3B - triple
    4B - homerun
    SO - strike out
    IPO - inplay out/reach on error

    Binary:
    0: causing an out or reaching on a fielding error or interference (SO or IPO)
    1: On base without causing an out

    Multi-class
    0: causing an out or reaching on a fielding error or interference (SO or IPO)
    1: walk, hit by pitch or single
    2: double
    3: triple
    4: homerun
    '''

    search_substrings = ["walks", "walked", "hit by pitch", "singles", "doubles", "triples", "homers", "strikes"]
    mapping = ["walk", "walk", "HBP", "1B", "2B", "3B", "HR", "SO"]

    #Mapping descriptions to outcome codes
    for substring, value in zip(search_substrings, mapping):
        # Check if the substring is present in the column
        mask = df['description'].str.contains(substring, case=False)
        # Assign the corresponding value to the 'result' column where the mask is True
        df.loc[mask, 'play_outcome'] = value

    df['play_outcome'] = df['play_outcome'].fillna("IPO")

    #Mapping outcomes to multi-class targets
    outcome_mapping = {"walk": 1,
                        "HBP": 1,
                        "1B": 1,
                        "2B": 2,
                        "3B": 3,
                        "HR": 4,
                        "SO": 0,
                        "IPO": 0}

    df["mc_target"] = df["play_outcome"].map(lambda x: outcome_mapping[x])
    df = df.drop(columns='play_outcome')

    #Mapping multi-class targets to binary targets, final target for model
    df["y_target"] = df["mc_target"].map(lambda y: 0 if y  == 0 else 1)

    #Setting at_bat id as index
    df = df.set_index("id")

    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:

    '''Data cleaning:
    cleaning nan values and dropping instances where weather conditions or pitch data was unrecorded
    '''

    #outs at start is nan in instances where there was a pitcher change at beginning of inning and the first at bat ended on first pitch, starting outs = 0:
    df["outs_at_start"] = df["outs_at_start"].fillna(0)

    df["pitch_class"] = df["pitch_location_zone"].apply(lambda x: 0 if x < 10 else 1)
    df["pitch_type_cat"] = df["pitch_type_des"].apply(lambda x: 0 if x == 'Four-Seam Fastball' or x == 'Slider' or x == 'Cutter' or x == 'Fastball' else 1)

    #droping rows without final pitch data (<0.001%)
    df = df.dropna(subset=["pitch_speed_mph", "pitch_location_zone"])

    return df


def merge_games_data(df: pd.DataFrame, filepath="raw_data/games_w_venue.csv") -> pd.DataFrame:
    '''
    provide at_bat dataset and a file path to games data, return merged DataFrame

    '''

    games = pd.read_csv(filepath)
    # Merging df and games
    games = games.rename(columns={"id": "game_id"})
    df = df.merge(games, how="left", on='game_id')

    return df

def merge_players_data(df: pd.DataFrame, filepath="raw_data/players.csv") -> pd.DataFrame:
    '''
    provide at_bat dataset and a file path to players data, return merged DataFrame

    '''

    players = pd.read_csv(filepath)

    # Merging df and hitters data
    hitters = players[~players.id.duplicated(keep="first")]
    hitters = hitters.add_prefix("hitter_")
    df = df.merge(hitters, how="left", on="hitter_id")

    # Merging df and pitchers data

    pitchers = players[~players.id.duplicated(keep="first")]
    pitchers = pitchers.add_prefix("pitcher_")
    df = df.merge(pitchers, how="left", on="pitcher_id")

    return df

def merge_teams_data(df: pd.DataFrame, filepath="raw_data/teams.csv") -> pd.DataFrame:
    '''
    provide at_bat dataset and a file path to teams data, return merged DataFrame

    '''

    teams = pd.read_csv(filepath)

    # Merging df and home team data
    home_team = teams.add_prefix("home_team_")
    df = df.rename(columns={"home_team": "home_team_id"})
    df = df.merge(home_team, how="left", on="home_team_id")

    # Merging df and away team data
    away_team = teams.add_prefix("away_team_")
    df = df.rename(columns={"away_team": "away_team_id"})
    df = df.merge(away_team, how="left", on="away_team_id")

    return df


def data_tuning(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Provide a DataFrame of merged data and return a dataset ready for feature engineering
    1. removed columns that are redundant or not used in modeling
    2. handles errored data points and corrects to logical inference of the data point value
    3. handles coverting dates to datetime
    4. Orders at bats oldest to newest
    '''

    #removing unwanted columns
    columns_to_remove = list(('description','scheduled',
                            'status', 'coverage', 'game_number',
                            'duration', 'double_header', 'entry_mode', 'reference',
                            'venue', 'home', 'away', 'broadcast', 'rescheduled','hitter_team_id', 'hitter_team_name','pitcher_position',
                            'pitcher_team_id', 'pitcher_team_name', 'home_team_name', 'home_team_market', 'home_team_abbr',
                            'away_team_name', 'away_team_market', 'away_team_abbr', 'pitch_location_zone'))

    df.drop(columns=columns_to_remove)

    #Cleaning up data points
    if 'outs_at_start' in df.columns:
        df['outs_at_start'] = df['outs_at_start'].apply(lambda x: 2 if x == 3 else x)

    if 'pitcher_pitch_count_at_bat_start' in df.columns:
        df['pitcher_pitch_count_at_bat_start'] = df['pitcher_pitch_count_at_bat_start'].apply(lambda x: 0 if x < 0 else x)

    #Coverting columns to the correct dtype
    df["at_bat_end_time"] = pd.to_datetime(df["at_bat_end_time"])

    df = df.sort_values(["at_bat_end_time"], ignore_index=True, ascending=True)

    return df


def write_feature_engineering_data_to_csv(df: pd.DataFrame, filepath="../../raw_data/all_ab_raw_data_w_target.csv"):
    '''
    save data ready for feature engineering to local drives
    '''

    df.to_csv(filepath)

    return f'.csv file save at {filepath}'


def create_dataset(ab_filepath='raw_data/plate_app_data/',
                    games_filepath="raw_data/games_w_venue.csv",
                    players_filepath='raw_data/players.csv',
                    teams_filepath="raw_data/teams.csv",
                    ) -> pd.DataFrame:
    '''
    Provide paths to tabular data, and return a finalized datset ready for feature enigeering in one function
    '''

    df = raw_data_parse(ab_filepath)
    df = set_targets(df)
    df = clean_data(df)
    df = merge_games_data(df, games_filepath)
    df = merge_players_data(df, players_filepath)
    df = merge_teams_data(df, teams_filepath)
    df = data_tuning(df)

    return df

def build_data_to_predict():
    """Build and save the pitchers.csv and hitters.csv from final_full_dataset and players tables
    Hitters and pitchers tables are used for the interface and make a predicrtion
    """

    # Import data
    data = pd.read_csv("mlb/interface/data/final_full_dataset.csv", index_col=0, parse_dates=["at_bat_end_time"])
    data = data.sort_values(by="at_bat_end_time", ascending=False)
    data["hitter_ab_count"] = data.groupby('hitter_id')['hitter_id'].transform('count')
    data["pitcher_ab_count"] = data.groupby('pitcher_id')['pitcher_id'].transform('count')

    # Import players
    players = pd.read_csv("mlb/interface/data/players.csv")
    players = players[~players.id.duplicated(keep="first")]

    # Create pitchers.csv
    pitchers = players[players.position == "P"][["id", "first_name", "last_name", "team_nickname", "primary_position"]]
    pitchers["full_name"] = pitchers.first_name + " " + pitchers.last_name
    pitchers = pitchers.drop(columns=["first_name", "last_name"])

    pitchers = pitchers.merge(data.drop_duplicates(subset='pitcher_id', keep='first')
                [["pitcher_id", "pitcher_hand", "pitcher_previous_stats_szn", "rolling_1pitch",
                    "rolling_3pitch", "rolling_10pitch", "pitcher_previous_stats_szn_bases",
                    "rolling_1pitch_bases", "rolling_3pitch_bases", "rolling_10pitch_bases",
                    "pitcher_strikes_spread", "pitcher_balls_spread", "pitcher_speed",
                    "pitcher_fast_spread", "pitcher_offspeed_spread", "pitcher_ab_count"]],
                how="left", left_on="id", right_on="pitcher_id")

    pitchers = pitchers.dropna().drop(columns="pitcher_id")
    pitchers.to_csv("mlb/interface/data/pitchers.csv")

    # Create hitters.csv
    hitters = players[players.position != "P"][["id", "first_name", "last_name", "team_nickname", "primary_position"]]
    hitters["full_name"] = hitters.first_name + " " + hitters.last_name
    hitters = hitters.drop(columns=["first_name", "last_name"])
    hitters = hitters[~hitters.full_name.duplicated(keep="first")]

    hitters = hitters.merge(data.drop_duplicates(subset='hitter_id', keep='first')
                            [["hitter_id", "hitter_hand", "hitter_position", "hitter_previous_stats_szn",
                            "rolling_1ab", "rolling_3ab", "rolling_10ab", "hitter_previous_stats_szn_slug",
                            "rolling_1ab_slug", "rolling_3ab_slug", "rolling_10ab_slug",
                            "hitter_strikes_eff", "hitter_balls_eff", "hitter_success_speed", "hitter_fast_eff",
                            "hitter_offspeed_eff", "hitter_ab_count"]],
                            how="left", left_on="id", right_on="hitter_id")

    hitters = hitters.dropna().drop(columns="hitter_id")
    hitters.to_csv("mlb/interface/data/hitters.csv")
