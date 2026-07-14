import streamlit as st
import pickle
import numpy as np

# Load Model
model = pickle.load(open("fare_model.pkl", "rb"))

# Load Scaler
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Taxi Fare Prediction using Linear Regression")

st.write("Enter Distance")

distance = st.number_input(
    "Distance (km)",
    min_value=0.0,
    value=1.0,
    step=0.1
)

if st.button("Predict Fare"):

    # Scaling
    distance_scaled = scaler.transform([[distance]])

    # Prediction
    prediction = model.predict(distance_scaled)

    st.success(f"Predicted Fare : {prediction[0]:.2f}")