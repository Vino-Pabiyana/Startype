import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ----------------------------
# Load trained models & encoders
# ----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("star_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")
    color_encoder = joblib.load("color_encoder.pkl")
    spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_models()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("⭐ Star Type Classification")
st.markdown("Predict the **type of a star** based on its characteristics.")

# Sidebar inputs
temperature = st.number_input("Temperature (K)", min_value=0, value=5000)
luminosity = st.number_input("Luminosity (L/Lo)", min_value=0.0, value=1.0)
radius = st.number_input("Radius (R/Ro)", min_value=0.0, value=1.0)
absolute_magnitude = st.number_input("Absolute Magnitude (Mv)", value=5.0)

color = st.selectbox("Color", color_encoder.classes_)
spectral_class = st.selectbox("Spectral Class", spectral_encoder.classes_)

# ----------------------------
# Prediction
# ----------------------------
if st.button("🔮 Predict Star Type"):
    # Prepare input as DataFrame
    input_df = pd.DataFrame({
        "Temperature": [temperature],
        "L": [luminosity],
        "R": [radius],
        "A_M": [absolute_magnitude],
        "Color": [color],
        "Spectral_Class": [spectral_class]
    })

    # Encode categorical variables
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])

    # Scale numerical values
    numeric_cols = ["Temperature", "L", "R", "A_M"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]

    st.success(f"🌟 Predicted Star Type: **{prediction}**")
