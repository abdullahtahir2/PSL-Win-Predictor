import streamlit as st
from PIL import Image
import plotly.express as px
import pickle
import pandas as pd

# Unpickling the trained model
try:
    xgbc_model = pickle.load(open("./PSL-Win-XGBC-model.pkl", "rb"))
except Exception as e:
    st.error(f"Error loading model: {e}")

# Title
st.markdown("<h1 style='color:Gold; text-align: center; font-size: 40px;'>Pakistan Super League (PSL) Win Predictor</h1>", unsafe_allow_html=True)

img = Image.open("./PSL-6.jpg")
st.image(img, width=700)

form = st.sidebar.form(key='my_form')
# Add a selectbox to the sidebar:
Team1 = form.selectbox(
    'Select Team Batting First',
    ('Islamabad United', 'Karachi Kings', 'Lahore Qalandars', 'Multan Sultans', 'Peshawar Zalmi', 'Quetta Gladiators')
)

Team2 = form.selectbox(
    'Select Team Batting Second',
    ('Karachi Kings', 'Islamabad United', 'Lahore Qalandars', 'Multan Sultans', 'Peshawar Zalmi', 'Quetta Gladiators')
)

# Target and other inputs
target = form.text_input('Target For the Team Batting Second', 110)
cur_runs = form.text_input('Current Runs of the Team Batting Second', 10)
wickets = form.text_input('Current Wickets of the Team Batting Second', 5.0)
overs = form.text_input('Current Overs Played by the Team Batting Second', 5.5)

submit_button = form.form_submit_button(label='Predict Win %')

if submit_button:
    try:
        # Convert inputs to floats
        target = float(target)
        cur_runs = float(cur_runs)
        wickets = float(wickets)
        overs = float(overs)

        # Create input data for prediction
        input_data = {
            "wickets": wickets,
            "balls_left": 120 - ((overs - overs % 1) * 6 + (overs % 1) * 10),
            "runs_left": target - cur_runs
        }
        input_data_df = pd.DataFrame(input_data, index=[0])

        # Predict probabilities
        prediction = xgbc_model.predict_proba(input_data_df)

        # Display prediction as a pie chart
        fig = px.pie(values=prediction[0], names=[Team1, Team2], title='Match Winning Percentage for Both Teams')
        st.plotly_chart(fig)

        # Display interpretation
        st.success(f"Interpretation: There is a {round(prediction[0][0] * 100)}% chance the team batting second ({Team2}) will lose (or the first team ({Team1}) will win) and a {round(prediction[0][1] * 100)}% chance that {Team2} will win.")
    
    except ValueError as e:
        st.error(f"Error in input: {e}")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
