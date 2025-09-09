import streamlit as st
import pandas as pd
import joblib
import base64

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
    layout="wide",
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
    /* Falling stars animation */
    @keyframes fall {
        0% {transform: translateY(-100vh);}
        100% {transform: translateY(100vh);}
    }
    .falling-star {
        position: fixed;
        top: 0;
        left: calc(100% * var(--pos));
        font-size: 25px;
        animation: fall linear infinite;
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
# Prediction Function
# -------------------------------
def predict_star(input_df):
    input_df["Color"] = color_encoder.transform(input_df["Color"])
    input_df["Spectral_Class"] = spectral_encoder.transform(input_df["Spectral_Class"])
    numeric_cols = ["Temperature", "L", "R", "A_M"]
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    prediction = model.predict(input_df)
    return prediction

star_types = {
    0: "Red Dwarf 🌟",
    1: "Brown Dwarf 🟤",
    2: "White Dwarf ⚪",
    3: "Main Sequence 🌞",
    4: "Supergiant 💫",
    5: "Hypergiant 🌌"
}

# -------------------------------
# Single Prediction
# -------------------------------
if st.sidebar.button("🔮 Predict Star Type"):
    input_df = pd.DataFrame([{
        "Temperature": temperature,
        "L": luminosity,
        "R": radius,
        "A_M": absolute_magnitude,
        "Color": color,
        "Spectral_Class": spectral_class
    }])
    prediction = predict_star(input_df)[0]
    st.subheader("🔭 Prediction Result")
    st.success(f"The predicted **Star Type** is: **{star_types[prediction]}**")

    # Falling stars animation
    falling_stars = "".join(
        f'<div class="falling-star" style="--pos:{i/10}; animation-duration:{3+i%3}s;">✨</div>'
        for i in range(10)
    )
    st.markdown(falling_stars, unsafe_allow_html=True)

# -------------------------------
# Batch Prediction via CSV Upload
# -------------------------------
st.divider()
st.subheader("📂 Batch Prediction (Upload CSV)")

uploaded_file = st.file_uploader("Upload CSV with star data", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.write("📄 Uploaded Data Preview:")
    st.dataframe(df.head())

    if st.button("🚀 Run Batch Prediction"):
        batch_df = df.copy()
        predictions = predict_star(batch_df)
        df["Predicted_Star_Type"] = [star_types[p] for p in predictions]

        st.success("✅ Batch Prediction Completed")
        st.dataframe(df)

        # Allow download
        csv = df.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="batch_star_predictions.csv">📥 Download Results</a>'
        st.markdown(href, unsafe_allow_html=True)

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
