# ===============================
# ⭐ Star Type Classification App
# ===============================

import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ----------------------------
# Load Models & Encoders
# ----------------------------
@st.cache(allow_output_mutation=True)
def load_models():
    # Load model
    model_path = "star_classifier_model.pkl"
    if os.path.exists(model_path):
        model = joblib.load(model_path)
    else:
        st.error(f"Missing file: {model_path}")
        st.stop()

    # Load scaler
    scaler_path = "scaler.pkl"
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)
    else:
        st.error(f"Missing file: {scaler_path}")
        st.stop()

    # Load or generate Color encoder
    color_path = "color_encoder.pkl"
    if os.path.exists(color_path):
        color_encoder = joblib.load(color_path)
    else:
        df = pd.read_csv("data/star_preprocessed.csv")
        color_encoder = LabelEncoder()
        color_encoder.fit(df["Color"])
        joblib.dump(color_encoder, color_path)

    # Load or generate Spectral_Class encoder
    spectral_path = "spectral_encoder.pkl"
    if os.path.exists(spectral_path):
        spectral_encoder = joblib.load(spectral_path)
    else:
        df = pd.read_csv("data/star_preprocessed.csv")
        spectral_encoder = LabelEncoder()
        spectral_encoder.fit(df["Spectral_Class"])
        joblib.dump(spectral_encoder, spectral_path)

    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_models()

# ----------------------------
# Streamlit UI
# ----------------------------
st.title("⭐ Star Type Classification")
st.markdown("Predict the **type of a star** based on its features.")

# Sidebar Inputs
st.sidebar.header("Enter Star Details")
temperature = st.sidebar.number_input("Temperature (K)", min_value=0, value=5000)
luminosity = st.sidebar.number_input("Luminosity (L/Lo)", min_value=0.0, value=1.0)
radius = st.sidebar.number_input("Radius (R/Ro)", min_value=0.0, value=1.0)
absolute_magnitude = st.sidebar.number_input("Absolute Magnitude (Mv)", value=5.0)
color = st.sidebar.selectbox("Color", color_encoder.classes_)
spectral_class = st.sidebar.selectbox("Spectral Class", spectral_encoder.classes_)

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

    # Encode categorical features
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])

    # Scale numeric features
    numeric_cols = ["Temperature", "L", "R", "A_M"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Make prediction
    prediction = model.predict(input_df)[0]

    # Optional: Map numeric prediction to star type name
    star_types = {
        0: "Brown Dwarf",
        1: "Red Dwarf",
        2: "White Dwarf",
        3: "Main Sequence",
        4: "Supergiant",
        5: "Hypergiant"
    }
    predicted_name = star_types.get(prediction, str(prediction))

    st.success(f"🌟 Predicted Star Type: **{predicted_name}**")
