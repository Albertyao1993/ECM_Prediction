import joblib
import os
import numpy as np
import pandas as pd

# Define the path to the saved model
saved_model_dir = 'saved_model'  # Replace with your actual path

model_path = os.path.join(saved_model_dir, 'svr_15min_heating.pkl')

# Load the model
svr_loaded = joblib.load(model_path)

# Define a single row of fake data
fake_data = {'is_holiday': 0,
             'day_of_week': 5, 'hour_of_day': 23, 'is_working_hour': 1, 'number_of_people': 0, 'Temperature': 12.9,
             'Humidity': 70.0, 'Dewpoint': 7.5, 'Sun Duration': 0.0, 'Precipitation Height': 0.0, 'Wind Speed': 4.0,
             'Wind Direction': 240.0, 'indoor_temperature': 12.9, 'temperature_difference': 0.0}

# Convert the fake data to a DataFrame
fake_data_df = pd.DataFrame([fake_data])

# Make a prediction
y_pred_fake = svr_loaded.predict(fake_data_df)

print(f'Prediction for fake data: {y_pred_fake[0]}')
