import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load Models and Encoders
# -------------------------------
@st.cache_resource
def load_models():
    with open("models/star_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    with open("color_encoder.pkl", "rb") as f:
        color_encoder = pickle.load(f)
    with open("spectral_encoder.pkl", "rb") as f:
        spectral_encoder = pickle.load(f)
    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_models()
import joblib
model = joblib.load("star_classifier_model.pkl")

# -------------------------------
# Streamlit UI
# -------------------------------
st.set_page_config(page_title="⭐ Star Type Classification", layout="centered")

st.title("⭐ Star Type Classification")
st.write("This app predicts the type of a star based on its physical characteristics.")

# Sidebar Inputs
st.sidebar.header("Star Features")
temperature = st.sidebar.number_input("Temperature (K)", min_value=2000, max_value=40000, value=5778)
luminosity = st.sidebar.number_input("Luminosity (L/Lo)", min_value=0.0001, max_value=100000.0, value=1.0, format="%.4f")
radius = st.sidebar.number_input("Radius (R/Ro)", min_value=0.1, max_value=1000.0, value=1.0, format="%.4f")
absolute_magnitude = st.sidebar.number_input("Absolute Magnitude (M)", min_value=-15.0, max_value=20.0, value=4.83)

color = st.sidebar.selectbox("Color", color_encoder.classes_)
spectral_class = st.sidebar.selectbox("Spectral Class", spectral_encoder.classes_)

# Prediction Button
if st.sidebar.button("🔮 Predict Star Type"):
    # Prepare data
    input_data = pd.DataFrame({
        "Temperature": [temperature],
        "L": [luminosity],
        "R": [radius],
        "A_M": [absolute_magnitude],
        "Color": [color],
        "Spectral_Class": [spectral_class]
    })

    # Preprocessing
    input_data["Color"] = color_encoder.transform(input_data["Color"])
    input_data["Spectral_Class"] = spectral_encoder.transform(input_data["Spectral_Class"])

    numeric_cols = ["Temperature", "L", "R", "A_M"]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Prediction
    prediction = model.predict(input_data)[0]

    st.subheader("🌌 Prediction Result")
    st.success(f"The predicted Star Type is: **{prediction}**")

