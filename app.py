import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Load Models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("star_classifier_model.pkl")
    scaler = joblib.load("scaler.pkl")
    color_encoder = joblib.load("color_encoder.pkl")
    spectral_encoder = joblib.load("spectral_encoder.pkl")
    return model, scaler, color_encoder, spectral_encoder

model, scaler, color_encoder, spectral_encoder = load_models()

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(
    page_title="⭐ Star Type Classifier",
    page_icon="✨",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom Styling
# -------------------------------
st.markdown(
    """
    <style>
    .main {
        background-color: #0f172a;
        color: white;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3 {
        color: #facc15;
    }
    .stButton>button {
        background-color: #facc15;
        color: black;
        font-weight: bold;
        border-radius: 10px;
        height: 3em;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #fde047;
        color: black;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------
# Title & Intro
# -------------------------------
st.title("⭐ Star Type Classification")
st.write(
    "A machine learning powered web app to classify **stars** into different types "
    "based on their physical properties. 🔭✨"
)

st.divider()

# -------------------------------
# Sidebar Inputs
# -------------------------------
st.sidebar.header("🔧 Input Star Details")

temperature = st.sidebar.number_input("Temperature (K)", min_value=2000, max_value=40000, value=5778)
luminosity = st.sidebar.number_input("Luminosity (L/Lo)", min_value=0.0, value=1.0)
radius = st.sidebar.number_input("Radius (R/Ro)", min_value=0.0, value=1.0)
absolute_magnitude = st.sidebar.number_input("Absolute Magnitude (Mv)", value=5.0)

color = st.sidebar.selectbox(
    "Color",
    color_encoder.classes_.tolist()
)

spectral_class = st.sidebar.selectbox(
    "Spectral Class",
    spectral_encoder.classes_.tolist()
)

# -------------------------------
# Prediction Button
# -------------------------------
if st.sidebar.button("🔮 Predict Star Type"):
    # Prepare input
    input_df = pd.DataFrame([{
        "Temperature": temperature,
        "L": luminosity,
        "R": radius,
        "A_M": absolute_magnitude,
        "Color": color,
        "Spectral_Class": spectral_class
    }])

    # Encode categorical features
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])

    # Scale numeric features
    numeric_cols = ["Temperature", "L", "R", "A_M"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

    # Predict
    prediction = model.predict(input_df)[0]

    # Star type mapping
    star_types = {
        0: "Red Dwarf 🌟",
        1: "Brown Dwarf 🟤",
        2: "White Dwarf ⚪",
        3: "Main Sequence 🌞",
        4: "Supergiant 💫",
        5: "Hypergiant 🌌"
    }

    st.subheader("🔭 Prediction Result")
    st.success(f"The predicted **Star Type** is: **{star_types[prediction]}**")

    st.balloons()

# -------------------------------
# Footer
# -------------------------------
st.divider()
st.markdown(
    """
    💡 **About this App**  
    This project demonstrates how machine learning can classify stars into types
    using their **temperature, luminosity, radius, absolute magnitude, color, and spectral class**.  
    Built with ❤️ using Streamlit.
    """
)
