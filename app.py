import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Configure page
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# App title and menu
st.markdown(
    """
    <style>
    .main-title {
        font-size: 36px;
        color: #4CAF50;
        text-align: center;
        margin-bottom: 10px;
    }
    .menu-bar {
        display: flex;
        justify-content: center;
        background-color: #4CAF50;
        padding: 10px;
        border-radius: 5px;
        margin-bottom: 20px;
    }
    .menu-item {
        margin: 0 15px;
        color: white;
        font-size: 16px;
        font-weight: bold;
        cursor: pointer;
    }
    .menu-item:hover {
        text-decoration: underline;
    }
    .sidebar-header {
        font-size: 20px;
        color: #333;
        font-weight: bold;
        margin-bottom: 15px;
    }
    .prediction-box {
        background-color: #d4edda;
        padding: 15px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .prediction-box h3 {
        color: #155724;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="main-title">🚗 Car Price Prediction App</div>', unsafe_allow_html=True)

menu_selection = st.markdown(
    """
    <div class="menu-bar">
        <span class="menu-item">🏠 Home</span>
        <span class="menu-item">📊 Insights</span>
        <span class="menu-item">📖 About</span>
        <span class="menu-item">👨‍💻 Developer</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Function to load model and scaler
def load_model():
    try:
        model = joblib.load("lasso_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("🚨 Error: Model or Scaler file not found. Ensure 'lasso_model.pkl' and 'scaler.pkl' are in the same directory.")
        st.stop()

# Sidebar for inputs
st.sidebar.markdown('<div class="sidebar-header">Enter Car Details</div>', unsafe_allow_html=True)

present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=100)
year = st.sidebar.number_input("Year of Purchase", min_value=2000, max_value=2023, step=1)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owners = st.sidebar.number_input("Number of Previous Owners", min_value=0, step=1)

# Mapping categorical inputs
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_map = {"Dealer": 0, "Individual": 1}
trans_map = {"Manual": 0, "Automatic": 1}

# Predict button
if st.sidebar.button("Predict Price"):
    model, scaler = load_model()

    # Prepare input for prediction
    features = np.array([
        present_price,
        kms_driven,
        2024 - year,
        fuel_map[fuel_type],
        seller_map[seller_type],
        trans_map[transmission],
        owners
    ]).reshape(1, -1)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Predict price
    predicted_price = model.predict(features_scaled)[0]

    # Format price with commas for thousands and proper units
    formatted_price = f"₹ {predicted_price:,.2f} Lakhs"
    st.markdown(
        f"""
        <div class="prediction-box">
            <h3>Predicted Selling Price</h3>
            <h1>{formatted_price}</h1>
        </div>
        """,
        unsafe_allow_html=True,
    )

# Footer section
st.sidebar.markdown("---")
st.sidebar.markdown("Developed by **Gurjap Singh** | Age 17")
st.sidebar.markdown("[LinkedIn Profile](https://www.linkedin.com/in/gurjap-singh-46696332a/)")
