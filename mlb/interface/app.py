import streamlit as st
import requests
import pandas as pd
import os

# Current path
path = os.getcwd()


# Define a dictionary of players and their corresponding teams
pitchers = pd.read_csv(f"{path}/mlb/interface/data/pitchers.csv", index_col=0)

hitters = pd.read_csv(f"{path}/mlb/interface/data/hitters.csv", index_col=0)


# Get a list of unique team names
unique_teams = sorted(list(pitchers.team_nickname.unique()))



# Create a Streamlit app
st.markdown(f"<h1 style='text-align:center'>MLB At Bat Predictor</h1>", unsafe_allow_html=True)


# Dropdown to select the pitching team
columns = st.columns(2)

pitching_team = columns[0].selectbox("Select Pitching Team", unique_teams)


# Get a list of players from the selected pitching team
if pitching_team:
    pitching_team_players = pitchers[pitchers.team_nickname == pitching_team]["full_name"].sort_values()
else:
    pitching_team_players = []


# Dropdown to select the pitcher from the pitching team
pitcher = columns[0].selectbox("Select Pitcher", pitching_team_players)


# Dropdown to select the hitting team
# hitting_team = unique_teams.remove(pitching_team)
hitting_team = columns[1].selectbox("Select Hitting Team", unique_teams)


# Get a list of players from the selected hitting team
if hitting_team:
    hitting_team_players = hitters[hitters.team_nickname == hitting_team]["full_name"].sort_values()
else:
    hitting_team_players = []


# Dropdown to select the hitter from the hitting team
hitter = columns[1].selectbox("Select Hitter", hitting_team_players)


# Display the selected teams and players
text_vs = f"{pitcher} ({pitchers[(pitchers.full_name == pitcher) & (pitchers.team_nickname == pitching_team)].primary_position.iloc[0]} / \
          {pitching_team}) VS {hitter} ({hitters[(hitters.full_name == hitter) & (hitters.team_nickname == hitting_team)].primary_position.iloc[0]} / \
          {hitting_team})"
st.write(f"<div style='text-align:center'><strong><span style='font-size:22px'>{text_vs}</span></strong></div>", unsafe_allow_html=True)


# Pitcher stats
columns = st.columns(2)

pitcher_stats = pitchers[(pitchers.full_name == pitcher) & (pitchers.team_nickname == pitching_team)]
pitcher_stats = pitcher_stats[['pitcher_ab_count', 'pitcher_hand', 'pitcher_previous_stats_szn', 'rolling_10pitch',
                              'pitcher_previous_stats_szn_bases', 'rolling_10pitch_bases', 'pitcher_fast_spread' ,'pitcher_offspeed_spread']]

pitcher_stats['pitcher_ab_count'] = int(pitcher_stats['pitcher_ab_count'])

pitcher_stats['pitcher_previous_stats_szn'] = round(pitcher_stats['pitcher_previous_stats_szn'],3)
pitcher_stats['rolling_10pitch'] = round(pitcher_stats['rolling_10pitch'],3)
pitcher_stats['pitcher_previous_stats_szn_bases'] = round(pitcher_stats['pitcher_previous_stats_szn_bases'],3)
pitcher_stats['rolling_10pitch_bases'] = round(pitcher_stats['rolling_10pitch_bases'],3)

pitcher_stats['pitcher_fast_spread'] = round(pitcher_stats['pitcher_fast_spread']*100, 2)
pitcher_stats['pitcher_offspeed_spread'] = round(pitcher_stats['pitcher_offspeed_spread']*100, 2)

pitcher_stats = pitcher_stats.rename(columns={'pitcher_ab_count': '2023 Batters Faced',
                              'pitcher_hand': 'Pitching Hand',
                              'pitcher_previous_stats_szn': 'Season Opp OBP',
                              'rolling_10pitch': 'Last 10 Batters Faced',
                              'pitcher_previous_stats_szn_bases': 'Season Opp Slugging',
                              'rolling_10pitch_bases': 'Last 10 Batters Slugging',
                              'pitcher_fast_spread': 'Fastball %',
                              'pitcher_offspeed_spread': 'Off-speed %'})

pitcher_stats = pitcher_stats.assign(hack='').set_index('hack').T

# columns[0].write(pitcher_stats)
columns[0].dataframe(pitcher_stats, width=1000)


# Hitter stats
hitter_stats = hitters[(hitters.full_name == hitter) & (hitters.team_nickname == hitting_team)]

hitter_stats = hitter_stats[['hitter_ab_count', 'hitter_hand', 'hitter_previous_stats_szn', 'rolling_10ab',
                              'hitter_previous_stats_szn_slug', 'rolling_10ab_slug', 'hitter_fast_eff' ,'hitter_offspeed_eff']]

hitter_stats['hitter_ab_count'] = int(hitter_stats['hitter_ab_count'])

hitter_stats['hitter_previous_stats_szn'] = round(hitter_stats['hitter_previous_stats_szn'],3)
hitter_stats['rolling_10ab'] = round(hitter_stats['rolling_10ab'],3)
hitter_stats['hitter_previous_stats_szn_slug'] = round(hitter_stats['hitter_previous_stats_szn_slug'],3)
hitter_stats['rolling_10ab_slug'] = round(hitter_stats['rolling_10ab_slug'],3)

hitter_stats['hitter_fast_eff'] = round(hitter_stats['hitter_fast_eff'], 3)
hitter_stats['hitter_offspeed_eff'] = round(hitter_stats['hitter_offspeed_eff'], 3)

hitter_stats = hitter_stats.rename(columns={'hitter_ab_count': '2023 At Bats',
                              'hitter_hand': 'Batter Hand',
                              'hitter_previous_stats_szn': 'Season OBP',
                              'rolling_10ab': 'Last 10 At Bats OBP',
                              'itter_previous_stats_szn_slug': 'Season Slugging',
                              'rolling_10ab_slug': 'Last 10 At Bats Slugging',
                              'hitter_fast_eff': 'Fastball Efficiency',
                              'hitter_offspeed_eff': 'Off-speed Efficiency'})



hitter_stats = hitter_stats.assign(hack='').set_index('hack').T

# columns[1].write(hitter_stats)
columns[1].dataframe(hitter_stats, width=1000)

# API
params = {
    "pitcher_name": pitcher,
    "hitter_name": hitter
}


if st.button("Predict"):

    # # # # Print resul
    # st.write(f"{hitter} bats the ball !!!")
    # win_gif = f"{path}/mlb/interface/illustrations/win.gif"
    # st.image(win_gif, width=500)

    # st.write("The pitcher was better")
    # lose_gif = f"{path}/mlb/interface/illustrations/lose.gif"
    # st.image(lose_gif, width=500)


    mbl_api_url = 'http://localhost:8000/predict'  # Replace with your API endpoint
    try:
        response = requests.get(mbl_api_url, params=params)
        response.raise_for_status()  # Raise an exception if the request is not successful

        prediction = response.json()

        pred = prediction.get('prediction')  # Use .get() to avoid KeyError if 'prediction' is missing
        proba = round(prediction.get('probability'),1) # Use .get() to avoid KeyError if 'probability' is missing

        if pred == 1:
            st.success("Batter wins this at bat!")
            st.success(f"Batter is projected to win this at bat, with a {proba} probability")
        elif pred == 0:
            st.error("Pitcher wins this at bat!")
            st.error(f"Pitcher is expected to win the at bat; batter has a {proba} probability to be successful")
        else:
            st.warning(f"Unexpected prediction value: {pred}")
    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while making the API request: {str(e)}")
    except KeyError:
        st.error("The API response is missing the 'prediction' key.")


st.write(pred)
st.write(f"{proba}%")
