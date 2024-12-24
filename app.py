import streamlit as st
import numpy as np
import joblib

# App title and configuration
st.set_page_config(page_title="Car Price Prediction", layout="wide", page_icon="🚗")
st.title("🚗 **Car Price Prediction App**")

# Function to load model and scaler
def load_model():
    try:
        model = joblib.load('lasso_model.pkl')  # Ensure the model file is in the same directory
        scaler = joblib.load('scaler.pkl')      # Ensure the scaler file is in the same directory
        return model, scaler
    except FileNotFoundError:
        st.error("🚨 **Error:** Model or Scaler file not found. Please ensure the files 'lasso_model.pkl' and 'scaler.pkl' are present.")
        st.stop()

# Sidebar for user input
with st.sidebar:
    st.header("📋 **Enter Car Details**")
    
    present_price = st.slider("💰 **Purchase Price** (in lakhs)", min_value=0.0, max_value=100.0, step=0.1, value=5.0)
    kms_driven = st.number_input("📏 **Kilometers Driven**", min_value=0, max_value=500000, step=100, value=10000)
    year = st.slider("📅 **Year of Purchase**", min_value=2000, max_value=2023, step=1, value=2015)
    fuel_type = st.selectbox("⛽ **Fuel Type**", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("🧑‍💼 **Seller Type**", ["Dealer", "Individual"])
    transmission = st.selectbox("⚙️ **Transmission**", ["Manual", "Automatic"])
    owners = st.slider("👨‍👩‍👧‍👦 **Number of Previous Owners**", min_value=0, max_value=5, step=1, value=0)

# Map categorical features
fuel_type_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_type_mapping = {"Dealer": 0, "Individual": 1}
transmission_mapping = {"Manual": 0, "Automatic": 1}

fuel_type_encoded = fuel_type_mapping[fuel_type]
seller_type_encoded = seller_type_mapping[seller_type]
transmission_encoded = transmission_mapping[transmission]

# Styling the layout
st.markdown(
    """
    <style>
    .predicted-price {
        background-color: #d4edda;
        padding: 20px;
        font-size: 24px;
        border-radius: 10px;
        color: #155724;
        text-align: center;
        font-weight: bold;
    }
    .sidebar .sidebar-content {
        background-color: #f4f4f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        font-size: 16px;
        padding: 10px 20px;
    }
    .stSlider > div {
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True
)

# Prediction Button
if st.sidebar.button("🚀 Predict Price"):
    model, scaler = load_model()
    
    # Prepare input data for prediction
    car_features = np.array([
        present_price,  # Purchase price
        kms_driven,
        2024 - year,  # Current year minus purchase year
        fuel_type_encoded,
        seller_type_encoded,
        transmission_encoded,
        owners
    ]).reshape(1, -1)

    # Scale the features using the scaler
    car_features_scaled = scaler.transform(car_features)
    
    # Predict the car price
    predicted_price = model.predict(car_features_scaled)[0]
    
    # Display the predicted price
    st.subheader("🔮 **Predicted Selling Price**")
    st.markdown(
        f"""
        <div class="predicted-price">
        💲 ₹ {predicted_price:,.2f} Lakhs
        </div>
        """, unsafe_allow_html=True
    )
    st.balloons()

# About the App and Developer sections
menu = st.sidebar.radio("📖 **Menu**", ["About App", "About Developer"])

if menu == "About App":
    st.header("📄 About the App")
    st.write(
        """
        This app predicts the **selling price** of used cars based on:
        - **Purchase price** of the car.
        - **Kilometers driven**.
        - **Year of purchase**.
        - **Fuel type**, **seller type**, and **transmission**.
        - **Number of previous owners**.

        The model used is a **Lasso Regression** trained on a car sales dataset.
        """
    )

elif menu == "About Developer":
    st.header("👨‍💻 About the Developer")
    st.write(
        """
        - **Name:** Gurjap Singh
        - **Age:** 17
        - **Enthusiast in AI and Machine Learning**
        - **[LinkedIn](https://www.linkedin.com/in/gurjap-singh-46696332a/)**
        """
    )
