import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and scaler
model = joblib.load('random_forest_model.pkl')
scaler = joblib.load('scaler.pkl')

# Set the page title and layout
st.set_page_config(page_title="Car Price Prediction", layout="wide")

# Create a function to format price
def format_price(price):
    lakhs = int(price)
    thousands = int((price - lakhs) * 100)
    
    if lakhs > 0:
        lakh_str = f"{lakhs} Lakh"
    else:
        lakh_str = ""
    
    if thousands > 0:
        thousand_str = f"{thousands} Thousand"
    else:
        thousand_str = ""
    
    return f"💲 ₹ {lakh_str} {thousand_str}".strip()

# Title of the app
st.title("Car Selling Price Prediction")

# Create a sidebar with the input menu on top
st.sidebar.header("Enter Car Details")

# Input fields for the user to provide data
present_price = st.sidebar.slider("💰 Purchase Price (in Lakhs)", 0.0, 100.0, 5.0, 0.1)
kms_driven = st.sidebar.slider("📏 Kilometers Driven", 1000, 200000, 10000)
year_of_purchase = st.sidebar.slider("📅 Year of Purchase", 2000, 2023, 2015)
fuel_type = st.sidebar.selectbox("⛽ Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("🧑‍💼 Seller Type", ["Individual", "Dealer"])
transmission = st.sidebar.selectbox("⚙️ Transmission", ["Manual", "Automatic"])
owner = st.sidebar.selectbox("👨‍👩‍👧‍👦 Number of Previous Owners", [0, 1, 2, 3, 4, 5])

# Convert categorical values to numeric as per your model's encoding
fuel_type_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}
transmission_map = {"Manual": 0, "Automatic": 1}
seller_type_map = {"Individual": 1, "Dealer": 0}

fuel_type_val = fuel_type_map[fuel_type]
transmission_val = transmission_map[transmission]
seller_type_val = seller_type_map[seller_type]

# Prepare the input for prediction
input_data = np.array([present_price, kms_driven, year_of_purchase, fuel_type_val, seller_type_val, transmission_val, owner]).reshape(1, -1)

# Scale the input using the scaler
input_scaled = scaler.transform(input_data)

# Make the prediction
predicted_price = model.predict(input_scaled)[0]

# Display the predicted price in the desired format
formatted_price = format_price(predicted_price)

# Show the prediction result
st.subheader(f"Predicted Selling Price: {formatted_price}")

# Add some more UI elements (optional) to improve the design
st.markdown("""
    <style>
    .stButton > button {
        background-color: #4CAF50;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        width: 100%;
    }
    .stSidebar {
        width: 300px;
    }
    </style>
    """, unsafe_allow_html=True)

# Footer (Optional) with a disclaimer
st.markdown("### Disclaimer: The predictions are based on historical data and may not reflect the current market conditions.")
