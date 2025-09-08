import streamlit as st
import pandas as pd
import joblib

# Title
st.title("Star Type Classification App ✨")
st.write("Predict the type of a star based on its features.")

# Load the trained model and scaler
model = joblib.load("star_classifier_model.pkl")
scaler = joblib.load("scaler.pkl")

# Load label encoders if you used them for categorical features
color_encoder = joblib.load("color_encoder.pkl")
spectral_encoder = joblib.load("spectral_encoder.pkl")

# Sidebar inputs
st.sidebar.header("Input Star Features")

Temperature = st.sidebar.number_input("Temperature")
L = st.sidebar.number_input("Luminosity (L)")
R = st.sidebar.number_input("Radius (R)")
A_M = st.sidebar.number_input("Absolute Magnitude (A_M)")

# For categorical features, use selectbox
Color = st.sidebar.selectbox(
    "Color",
    options=color_encoder.classes_
)

Spectral_Class = st.sidebar.selectbox(
    "Spectral Class",
    options=spectral_encoder.classes_
)

# Create input DataFrame
input_data = pd.DataFrame({
    "Temperature": [Temperature],
    "L": [L],
    "R": [R],
    "A_M": [A_M],
    "Color": [Color],
    "Spectral_Class": [Spectral_Class]
})

# Encode categorical columns
input_data["Color"] = color_encoder.transform(input_data["Color"])
input_data["Spectral_Class"] = spectral_encoder.transform(input_data["Spectral_Class"])

# Scale numeric columns
numeric_cols = ["Temperature", "L", "R", "A_M"]
input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

# Predict button
if st.button("Predict Star Type"):
    prediction = model.predict(input_data)
    st.success(f"Predicted Star Type: {prediction[0]}")
