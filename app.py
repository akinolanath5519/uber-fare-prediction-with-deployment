import streamlit as st
import numpy as np
import pickle

# Load model and scaler
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Uber Fare Prediction 🚕")
st.write("Enter ride details to predict fare amount.")

# Inputs
pickup_longitude = st.number_input("Pickup Longitude")
pickup_latitude = st.number_input("Pickup Latitude")
dropoff_longitude = st.number_input("Dropoff Longitude")
dropoff_latitude = st.number_input("Dropoff Latitude")
passenger_count = st.number_input("Passenger Count", min_value=1, max_value=6, step=1)
pickup_hour = st.number_input("Pickup Hour (0–23)", min_value=0, max_value=23, step=1)
pickup_day = st.number_input("Pickup Day of Week (0=Mon, 6=Sun)", min_value=0, max_value=6, step=1)

if st.button("Predict Fare"):
    features = np.array([[pickup_longitude, pickup_latitude,
                          dropoff_longitude, dropoff_latitude,
                          passenger_count, pickup_hour, pickup_day]])
    
    scaled = scaler.transform(features)
    prediction = model.predict(scaled)[0]

    st.success(f"Estimated Fare: **${prediction:.2f}**")
