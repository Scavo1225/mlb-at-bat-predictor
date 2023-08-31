import pandas as pd
import requests
import json
import time
import datetime as dt
import numpy as np
import os


def raw_data_parse(filepath='../raw_data/plate_app_data/') -> pd.DataFrame:

    '''Provide a directory to JSON files from API call source Sport Radar - Play by Play
    Parsing through JSON to pull the key data into a dataframe in raw form for all at bats in the game set

    '''


    directory = filepath

    for filename in os.listdir(directory):
    f = os.path.join(directory, filename)
    with open(f) as user_file:
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
                inning = gdata["game"]["innings"][ing]["number"]
                side = gdata["game"]["innings"][ing]["halfs"][hlf]["half"]

                hitter_id = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["hitter_id"]
                hitter_hand = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["hitter_hand"]

                pitcher_id = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["pitcher_id"]
                pitcher_hand = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["pitcher_hand"]

                if 'description' not in gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]:
                    continue

                description = gdata["game"]["innings"][ing]["halfs"][hlf]["events"][atb]["at_bat"]["description"]

                if 'weather' in gdata["game"]["innings"][ing]["halfs"][hlf]:
                    if "temp_f" not in gdata["game"]["innings"][ing]["halfs"][hlf]["weather"]["current_conditions"]:
                        temp_f = np.nan
                    else:
                        temp_f = gdata["game"]["innings"][ing]["halfs"][hlf]["weather"]["current_conditions"]["temp_f"]

                    if "condition" not in gdata["game"]["innings"][ing]["halfs"][hlf]["weather"]["current_conditions"]:
                        weather_condition = np.nan
                    else:
                        weather_condition = gdata["game"]["innings"][ing]["halfs"][hlf]["weather"]["current_conditions"]["condition"]

                    if "humidity" not in gdata["game"]["innings"][ing]["halfs"][hlf]["weather"]["current_conditions"]:
                        humidity  = np.nan
                    else:
                        humidity = gdata["game"]["innings"][ing]["halfs"][hlf]["weather"]["current_conditions"]["humidity"]

                    if "wind" not in gdata["game"]["innings"][ing]["halfs"][hlf]["weather"]["current_conditions"]:
                        wind_speed_mph  = np.nan

                    else:
                        wind_speed_mph = gdata["game"]["innings"][ing]["halfs"][hlf]["weather"]["current_conditions"]["wind"]["speed_mph"]

                else:
                    temp_f = np.nan
                    weather_condition = np.nan
                    humidity = np.nan
                    wind_speed_mph = np.nan

                try:
                    at_bat_end_time = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["wall_clock"]["end_time"]
                except: at_bat_end_time = np.nan

                try:
                    pitch_location_zone = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["mlb_pitch_data"]["zone"]
                except:
                    pitch_location_zone = np.nan

                try:
                    pitch_type_code = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["mlb_pitch_data"]["code"]
                except:
                    pitch_type_code = np.nan

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

                try:
                    output_code = gdata["game"]['innings'][ing]['halfs'][hlf]['events'][atb]["at_bat"]["events"][-1]["outcome_id"]
                except:
                    output_code = np.nan

                data = {'id': id,
                        'game_id': game_id,
                        'inning': inning,
                        'side': side,
                        'hitter_id': hitter_id,
                        'hitter_hand' :hitter_hand,
                        'pitcher_id': pitcher_id,
                        'pitcher_hand': pitcher_hand,
                        'description': description,
                        'temp_f': temp_f,
                        'weather_condition': weather_condition,
                        'humidity': humidity,
                        'wind_speed_mph': wind_speed_mph,
                        'at_bat_end_time': at_bat_end_time,
                        'pitch_location_zone': pitch_location_zone,
                        'pitch_type_code': pitch_type_code,
                        'pitch_type_des': pitch_type_des,
                        'pitch_speed_mph': pitch_speed_mph,
                        'pitch_count_at_bat': pitch_count_at_bat,
                        'pitcher_pitch_count_at_bat_start': pitcher_pitch_count_at_bat_start,
                        'outs_at_start': outs_at_start,
                        'output_code': output_code
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

    #Mapping multi-class targets to binary targets, final target for model
    df["y_target"] = df["mc_target"].map(lambda y: 0 if y  == 0 else 1)

    #Setting at_bat id as index
    df = df.set_index("id")

    return df

def clean_data(df: pd.DateFrame) -> pd.DataFrame:

    '''Data cleaning:
    cleaning nan values and dropping instances where weather conditions or pitch data was unrecorded
    '''

    #outs at start is nan in instances where there was a pitcher change at beginning of inning and the first at bat ended on first pitch, starting outs = 0:
    df["outs_at_start"] = df["outs_at_start"].fillna(0)

    #weather data missing is dropped from data set (<0.001% of at bats)
    df = df.dropna(subset=["weather_condition"])

    #droping rows without final pitch data (<0.001%)
    df = df.dropna(subset=["pitch_type_code", "pitch_speed_mph"])

    return df

def write_raw_data_to_csv(df: pd.DataFrame, filepath="../raw_data/all_ab_raw_data_w_target.csv") -> .csv:
        '''
    save data ready for merging to local drives
    '''

    df.to_csv(filepath)

    return None


def merge_games_data(df: pd.DataFrame, filepath="../raw_data/games_w_venue.csv") -> pd.DataFrame:
    '''
    provide at_bat dataset and a file path to games data, return merged DataFrame

    '''

    games = pd.read_csv(filepath)
    # Merging df and games
    games = games.rename(columns={"id": "game_id"})
    df = df.merge(games, how="left", on='game_id')

    return df

def merge_players_data(df: pd.DataFrame, filepath="../raw_data/players.csv") -> pd.DataFrame:
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

def merge_teams_data(df: pd.DataFrame, filepath="../raw_data/teams.csv") -> pd.DataFrame:
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

def merge_stadiums_data(df: pd.DataFrame, filepath="../raw_data/stadiums.csv") -> pd.DataFrame:
    '''
    provide at_bat dataset and a file path to stadiums data, return merged DataFrame

    '''

    stadiums = pd.read_csv(filepath)

    # Merging df and stadiums data
    venue = stadiums.add_prefix("stadium_")
    df = df.rename(columns={"venue_id": "stadium_id"})
    df = df.merge(venue, how="left", on="stadium_id")

    away_stadium = stadiums
    away_stadium["abbr"] = teams.abbr
    away_stadium = away_stadium.add_prefix("away_stadium_")
    away_stadium = away_stadium.rename(columns={"away_stadium_abbr": "away_team_abbr"})
    df = df.merge(away_stadium, how="left", on="away_team_abbr")

    return df


def data_tuning(df: pd.DataFrame, columns=None: list) -> pd.DataFrame:
    '''
    Provide a DataFrame of merged data and return a dataset ready for feature engineering

    Steps:
    Provide a listing of columns to removed from merged data set.
       if not list is passed to function, and default listing columns are removed

    Also handles:
    errored data points and corrects to logical inference of the data point value
    coverting dates to datetime


    '''
    try:
    if columns = None:
        columns = list(('description', 'play_outcome', 'mc_target',
                            'Unnamed: 0', 'status', 'coverage', 'game_number',
                            'duration', 'double_header', 'entry_mode', 'reference',
                            'venue', 'home', 'away', 'broadcast', 'rescheduled','hitter_team_id', 'hitter_team_name','pitcher_position',
                            'pitcher_team_id', 'pitcher_team_name', 'home_team_name', 'home_team_market', 'home_team_abbr',
                            'away_team_name', 'away_team_market', 'away_team_abbr', 'stadium_name', 'stadium_market', 'stadium_surface', 'stadium_address',
                            'stadium_city', 'stadium_state', 'stadium_zip', 'stadium_country', 'stadium_field_orientation', 'stadium_time_zone', 'away_stadium_id',
                            'away_stadium_name', 'away_stadium_market', 'away_stadium_surface', 'away_stadium_address', 'away_stadium_city', 'away_stadium_state', 'away_stadium_zip',
                            'away_stadium_country', 'away_stadium_field_orientation', 'away_stadium_stadium_type', 'away_stadium_time_zone', 'pitch_type_des'))

        df.drop(columns=columns)

        #Cleaning up data points
        if 'outs_at_start' in df.columns:
            df['outs_at_start'] = data['outs_at_start'].apply(lambda x: 2 if x == 3 else x)

        if 'pitcher_pitch_count_at_bat_start' in df.columns:
            df['pitcher_pitch_count_at_bat_start'] = data['pitcher_pitch_count_at_bat_start'].apply(lambda x: 0 if x < 0 else x)

        if 'wind_speed_mph' in df.columns:
            df['wind_speed_mph'] = data['wind_speed_mph'].apply(lambda x: 50 if x > 50 else x)

        #Coverting columns to the correct dtype
        df["scheduled"] = pd.to_datetime(df["scheduled"])
        df["at_bat_end_time"] = pd.to_datetime(df["at_bat_end_time"])

        df = df.sort_values(["at_bat_end_time", "inning"], ignore_index=True, ascending=True)

        return df

    except:
        return "Ensure to keep at_bat_end_time and inning in the dataset columns for ordering. Please try again"

def write_feature_engineering_data_to_csv(df: pd.DataFrame, filepath="../raw_data/all_ab_raw_data_w_target.csv") -> .csv:
    '''
    save data ready for feature engineering to local drives
    '''

    df.to_csv(filepath)

    return None
