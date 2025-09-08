import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -----------------------------
# Load Models & Encoders
# -----------------------------
@st.cache_resource
def load_models():
    model = joblib.load("star_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")
    color_encoder = joblib.load("color_encoder.pkl")
    spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_models()

# -----------------------------
# Sidebar: Input Form
# -----------------------------
st.sidebar.header("🌟 Enter Star Details")

temperature = st.sidebar.number_input("Temperature (K)", min_value=2000, max_value=50000, step=1000)
luminosity = st.sidebar.number_input("Luminosity (L/Lo)", min_value=0.0001, max_value=100000.0, step=0.1, format="%.4f")
radius = st.sidebar.number_input("Radius (R/Ro)", min_value=0.01, max_value=1000.0, step=0.01, format="%.2f")
absolute_magnitude = st.sidebar.number_input("Absolute Magnitude (M)", min_value=-15.0, max_value=20.0, step=0.1)

color = st.sidebar.selectbox("Color", ["Red", "Blue", "White", "Yellow", "Orange", "Other"])
spectral_class = st.sidebar.selectbox("Spectral Class", ["O", "B", "A", "F", "G", "K", "M"])

# -----------------------------
# Prediction Button
# -----------------------------
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

    # Encode categorical features
    input_data["Color"] = color_encoder.transform(input_data["Color"])
    input_data["Spectral_Class"] = spectral_encoder.transform(input_data["Spectral_Class"])

    # Scale numeric features
    numeric_cols = ["Temperature", "L", "R", "A_M"]
    input_data[numeric_cols] = scaler.transform(input_data[numeric_cols])

    # Predict
    prediction = model.predict(input_data)[0]

    st.success(f"⭐ Predicted Star Type: **{prediction}**")

# -----------------------------
# Main Section
# -----------------------------
st.title("⭐ Star Type Classification App")
st.write("""
This app predicts the **type of a star** based on its properties:
- Temperature  
- Luminosity  
- Radius  
- Absolute Magnitude  
- Color  
- Spectral Class  
""")
