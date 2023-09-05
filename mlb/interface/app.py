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
unique_teams = list(pitchers.team_nickname.unique())


# Create a Streamlit app
st.markdown(f"<h1 style='text-align:center'>MLB Game Predictor - OneHitWonder</h1>", unsafe_allow_html=True)


# Dropdown to select the pitching team
columns = st.columns(2)

pitching_team = columns[0].selectbox("Select Pitching Team", unique_teams)


# Get a list of players from the selected pitching team
if pitching_team:
    pitching_team_players = pitchers[pitchers.team_nickname == pitching_team]["full_name"]
else:
    pitching_team_players = []


# Dropdown to select the pitcher from the pitching team
pitcher = columns[0].selectbox("Select Pitcher", pitching_team_players)


# Dropdown to select the hitting team
hitting_team = columns[1].selectbox("Select Hitting Team", unique_teams)


# Get a list of players from the selected hitting team
if hitting_team:
    hitting_team_players = hitters[hitters.team_nickname == hitting_team]["full_name"]
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
pitcher_stats = pitcher_stats.assign(hack='').set_index('hack').T

columns[0].write(pitcher_stats)


# Hitter stats
hitter_stats = hitters[(hitters.full_name == hitter) & (hitters.team_nickname == hitting_team)]
hitter_stats = hitter_stats.assign(hack='').set_index('hack').T

columns[1].write(hitter_stats)


# API
params = {
    "pitcher_name": pitcher,
    "hitter_name": hitter
}

mbl_api_url = ''  # Replace with your API endpoint
try:
    response = requests.get(mbl_api_url, params=params)
    response.raise_for_status()  # Raise an exception if the request is not successful

    prediction = response.json()
    pred = prediction.get('y_target')  # Use .get() to avoid KeyError if 'y_target' is missing

    if pred == 1:
        st.success('The Hitter is going to get at least one Base.')
    elif pred == 0:
        st.error('The Hitter is not going to get on a Base.')
    else:
        st.warning(f"Unexpected prediction value: {pred}")
except requests.exceptions.RequestException as e:
    st.error(f"An error occurred while making the API request: {str(e)}")
except KeyError:
    st.error("The API response is missing the 'y_target' key.")


# Generate a prediction

if st.button("Predict"):

    # # # Print resul
    st.write(f"{hitter} bats the ball !!!")
    win_gif = f"{path}/mlb/interface/illustrations/win.gif"
    st.image(win_gif, width=500)

    st.write("The pitcher was better")
    lose_gif = f"{path}/mlb/interface/illustrations/lose.gif"
    st.image(lose_gif, width=500)
