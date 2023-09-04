import streamlit as st
import requests

# Define a dictionary of players and their corresponding teams
player_teams = {
    "Player1": "TeamA",
    "Player2": "TeamA",
    "Player3": "TeamB",
    "Player4": "TeamB",
}

# Get a list of unique team names
unique_teams = list(set(player_teams.values()))

# Create a Streamlit app
st.title("MLB Game Predictor - OneHitWonder ")

# Dropdown to select the pitching team
pitching_team = st.selectbox("Select Pitching Team", unique_teams)

# Get a list of players from the selected pitching team
if pitching_team:
    pitching_team_players = [player for player, team in player_teams.items() if team == pitching_team]
else:
    pitching_team_players = []

# Dropdown to select the pitcher from the pitching team
pitcher = st.selectbox("Select Pitcher", pitching_team_players)

# Dropdown to select the hitting team
hitting_team = st.selectbox("Select Hitting Team", unique_teams)

# Get a list of players from the selected hitting team
if hitting_team:
    hitting_team_players = [player for player, team in player_teams.items() if team == hitting_team]
else:
    hitting_team_players = []

# Dropdown to select the hitter from the hitting team
hitter = st.selectbox("Select Hitter", hitting_team_players)

# Display the selected teams and players
st.write(f"Pitching Team: {pitching_team}")
if pitcher:
    st.write(f"Pitcher: {pitcher}")

st.write(f"Hitting Team: {hitting_team}")
if hitter:
    st.write(f"Hitter: {hitter}")

params = {
    "pitcher_name": pitcher,
    "hitter_name": hitter
}

mbl_api_url = 'XXXXXXXXXXXXXXXXXXXXXXXX'  # Replace with your API endpoint
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
