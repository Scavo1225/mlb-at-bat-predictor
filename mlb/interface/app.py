import streamlit as st
import requests


with st.form(key='params_for_api'):
    pitcher_name = st.text_input('Pitcher Name')
    hitter_name = st.text_input('Hitter Name')
    location = st.text_input('Location')
    at_bats = st.number_input('Number of At-Bats', min_value=0, step=1, value=0)

    st.form_submit_button('Make prediction')

params = dict(
    pitcher_name=pitcher_name,
    hitter_name=hitter_name,
    location=location,
    at_bats=at_bats
)


mbl_api_url = 'XXXXXXXXXXXXXXXXXXXXXXXX'
response = requests.get(mbl_api_url_api_url, params=params)

prediction = response.json()

pred = prediction['y_target']

if pred == 1:
    st.success('The Hitter is going to get at least one Base.')
elif pred == 0:
    st.error('The Hitter is not going to get on a base.')
