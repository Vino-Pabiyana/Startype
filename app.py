import os
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

@st.cache_resource
def load_models():
    # Load trained model
    model = joblib.load("star_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")

    # Check and load/create Color encoder
    if os.path.exists("color_encoder.pkl"):
        color_encoder = joblib.load("color_encoder.pkl")
    else:
        df = pd.read_csv("data/star_preprocessed.csv")
        color_encoder = LabelEncoder()
        color_encoder.fit(df["Color"])
        joblib.dump(color_encoder, "color_encoder.pkl")

    # Check and load/create Spectral encoder
    if os.path.exists("spectral_encoder.pkl"):
        spectral_encoder = joblib.load("spectral_encoder.pkl")
    else:
        df = pd.read_csv("data/star_preprocessed.csv")
        spectral_encoder = LabelEncoder()
        spectral_encoder.fit(df["Spectral_Class"])
        joblib.dump(spectral_encoder, "spectral_encoder.pkl")

    return model, scaler, color_encoder, spectral_encoder
