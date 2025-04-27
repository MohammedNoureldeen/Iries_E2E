# src/app.py
import streamlit as st
import joblib
import pandas as pd
import numpy as np

# Load the model and scaler
model = joblib.load('data/model.joblib')
scaler = joblib.load('data/scaler.joblib')

# Set the title of the app
st.title('Iris Species Prediction')

# Collect user input for the features
sepal_length = st.number_input('Sepal Length (cm):', min_value=0.0, step=0.1)
sepal_width = st.number_input('Sepal Width (cm):', min_value=0.0, step=0.1)
petal_length = st.number_input('Petal Length (cm):', min_value=0.0, step=0.1)
petal_width = st.number_input('Petal Width (cm):', min_value=0.0, step=0.1)

# Button to trigger the prediction
if st.button('Predict'):
    try:
        # Create features array and scale
        features = np.array([sepal_length, sepal_width, petal_length, petal_width]).reshape(1, -1)
        features_scaled = scaler.transform(features)

        # Make prediction
        prediction = model.predict(features_scaled)[0]

        # Display the result
        st.success(f'Predicted Species: {prediction}')
    except Exception as e:
        st.error(f'Error: {str(e)}')
