import streamlit as st
import joblib
import numpy as np

# -------------------------------
# Page Config
# -------------------------------
st.set_page_config(page_title="Laptop Price Predictor", layout="centered")

st.title("💻 Laptop Price Predictor 💻")
st.markdown("Predict laptop prices based on specifications using Machine Learning")

# -------------------------------
# Custom Styling (Outline Red Button + Red Price)
# -------------------------------
st.markdown("""
    <style>
    /* Button - Red Outline */
    div.stButton > button {
        font-size: 18px;
        padding: 12px 24px;
        border-radius: 10px;
        background-color: transparent;
        color: #E53935;
        font-weight: 600;
        border: 2px solid #E53935;
    }

    /* Button Hover */
    div.stButton > button:hover {
        background-color: #E53935;
        color: white;
    }

    /* Price Text - Red */
    .price-text {
        font-size: 26px;
        font-weight: bold;
        color: #FF5252;
        text-align: center;
        margin-top: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model & Encoders
# -------------------------------
@st.cache_resource
def load_files():
    model = joblib.load("laptop_price_model.pkl")
    encoders = joblib.load("label_encoders.pkl")
    return model, encoders

try:
    model, encoders = load_files()
except Exception as e:
    st.error(f"Error loading model files: {e}")
    st.stop()

# -------------------------------
# Model Info
# -------------------------------
st.info("Model: Random Forest | Performance: R² ≈ 0.93")

# -------------------------------
# Extract Encoders
# -------------------------------
processor_encoder = encoders['Processor_Model']
ram_encoder = encoders['RAM']
os_encoder = encoders['Operating_System']
graphics_encoder = encoders['Graphics_Name']
warranty_encoder = encoders['Warranty']

# -------------------------------
# User Inputs
# -------------------------------
st.subheader("Enter Laptop Specifications")

col1, col2 = st.columns(2)

with col1:
    processor = st.selectbox("Processor Model", processor_encoder.classes_)
    ram = st.selectbox("RAM", ram_encoder.classes_)
    os = st.selectbox("Operating System", os_encoder.classes_)

with col2:
    graphics = st.selectbox("Graphics", graphics_encoder.classes_)
    warranty = st.selectbox("Warranty", warranty_encoder.classes_)

display_size = st.slider("Display Size (in inches)", 10.0, 20.0, 15.6)
storage = st.slider("Storage (GB)", 64, 4000, 512)

# -------------------------------
# Prediction
# -------------------------------
if st.button(" Predict Laptop Price"):
    try:
        input_data = np.array([[
            processor_encoder.transform([processor])[0],
            ram_encoder.transform([ram])[0],
            os_encoder.transform([os])[0],
            graphics_encoder.transform([graphics])[0],
            display_size,
            storage,
            warranty_encoder.transform([warranty])[0]
        ]])

        prediction = model.predict(input_data)[0]

        st.markdown(
            f"<div class='price-text'> Estimated Laptop Price: ₹ {prediction:,.0f}</div>",
            unsafe_allow_html=True
        )

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; font-size:12px; color:#A0A0A0;'>Developed by Purvaja Dodke | Machine Learning Project</p>",
    unsafe_allow_html=True
)
