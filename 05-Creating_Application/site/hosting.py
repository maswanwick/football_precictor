#########################
# Import the dependencies
#########################
import pandas as pd
import json
from flask import Flask, render_template, request
import pickle
import numpy as np

#########################
# Helper functions
#########################

# load csv into dataframe
def getReferenceDataFrameAsJSON(ref, subcat=''):
    ref_df = pd.read_csv(f'../Reference_Data/{ref}_Data.csv')
    ref_df = ref_df.rename(columns={"Unnamed: 0": f"{ref}_ID"})
    if (subcat != ''):
        ref_df = ref_df.rename(columns={f"{ref}": f"{subcat}{ref}"})
    json_data = ref_df.to_json(orient='records')
    return json.loads(json_data)

# build the feature dataframe and pre-fill with 0s
def getStarterPredictionDataFrame():
    prediction_df = pd.read_csv('../Building_Model_Exports/Feature_Names.csv')
    temp_df = pd.DataFrame(0, index=np.arange(1), columns=prediction_df.columns)
    prediction_df = pd.concat([prediction_df,temp_df])
    return prediction_df

# turn HH:mm into seconds as a float
def getSecondsFromTime(time_string):
    time_parts = time_string.split(':')
    minutes = int(time_parts[1]) / 60
    hours = int(time_parts[0]) + minutes
    seconds = hours * 60 * 60
    return float(seconds)

# update the feature dataframe with selected feature data
def populateSelectedFeatures(json_object, prediction_df):
    prediction_df[f"Home Team_{json_object['home_team']}"] = 1
    prediction_df["Home Team Pre-Game Season W/L Ratio"] = float(json_object['home_win_pct'])
    prediction_df["Home Team Pre-Game Season W Streak"] = int(json_object['home_win_streak'])
    prediction_df["Home Team Pre-Game Season L Streak"] = int(json_object['home_loss_streak'])
    prediction_df["Home Team Pre-Game Season Avg Points For"] = float(json_object['home_avg_pts_for'])
    prediction_df["Home Team Pre-Game Season Avg Points Against"] = float(json_object['home_avg_pts_against'])
    prediction_df[f"Away Team_{json_object['away_team']}"] = 1
    prediction_df["Away Team Pre-Game Season W/L Ratio"] = float(json_object['away_win_pct'])
    prediction_df["Away Team Pre-Game Season W Streak"] = int(json_object['away_win_streak'])
    prediction_df["Away Team Pre-Game Season L Streak"] = int(json_object['away_loss_streak'])
    prediction_df["Away Team Pre-Game Season Avg Points For"] = float(json_object['away_avg_pts_for'])
    prediction_df["Away Team Pre-Game Season Avg Points Against"] = float(json_object['away_avg_pts_against'])
    prediction_df[f"Venue_{json_object['venue']}"] = 1
    prediction_df[f"City_{json_object['city']}"] = 1
    prediction_df[f"State_{json_object['state']}"] = 1
    prediction_df[f"Month_{json_object['month']}"] = 1
    prediction_df["Season Week"] = int(json_object['week_num'])
    prediction_df["Time (EST)"] = getSecondsFromTime(json_object['time'])
    prediction_df[f"Weather Condition_{json_object['weather']}"] = 1
    prediction_df["Temperature (F)"] = float(json_object['temp'])
    return prediction_df

# load the trained model and scaler and predict the outcome
def getPrediction(prediction_df):
    test_scaler = pickle.load(open("../Building_Model_Exports/scaler_model","rb"))
    test_rf = pickle.load(open("../Building_Model_Exports/scaled_knn_model","rb"))
    prediction_features = prediction_df.values
    prediction_scaled_features = test_scaler.transform(prediction_features)
    return str(test_rf.predict(prediction_scaled_features)[0])

#########################
# Setup the Flask app
#########################
hosting = Flask(__name__)

#########################
# Base route
#########################
@hosting.route("/")
def home():
    return render_template('wizard.html');

#########################
# Dropdown data source routes
#########################
@hosting.route("/api/v1.0/cities")
def getCities():
    return getReferenceDataFrameAsJSON('Cities')

@hosting.route("/api/v1.0/months")
def getMonths():
    return getReferenceDataFrameAsJSON('Months')

@hosting.route("/api/v1.0/states")
def getStates():
    return getReferenceDataFrameAsJSON('States')

@hosting.route("/api/v1.0/times")
def getTimes():
    return getReferenceDataFrameAsJSON('Times')

@hosting.route("/api/v1.0/venues")
def getVenues():
    return getReferenceDataFrameAsJSON('Venues')

@hosting.route("/api/v1.0/weather")
def getWeather():
    return getReferenceDataFrameAsJSON('Weather')

@hosting.route("/api/v1.0/hometeams")
def getHomeTeams():
    return getReferenceDataFrameAsJSON('Teams', 'Home')

@hosting.route("/api/v1.0/awayteams")
def getAwayTeams():
    return getReferenceDataFrameAsJSON('Teams', 'Away')

#########################
# Route used to 
#########################
@hosting.route("/api/v1.0/make_prediction", methods=['POST'])
def makePredition():
    form = json.loads(request.form['data'])

    prediction_df = getStarterPredictionDataFrame()
    prediction_df = populateSelectedFeatures(form, prediction_df)
    outcome = getPrediction(prediction_df)

    winning_team = "nfl_logo"

    if outcome == "1":
        winning_team = form['home_team']
    elif outcome == "2":
        winning_team = form['away_team']
    elif outcome == "3":
        winning_team = "tie"     

    matchup_results = {
        "winning_team": winning_team
    }

    return json.dumps(matchup_results)

if __name__ == '__main__':
    hosting.run(port=8000, debug=True)