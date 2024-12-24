import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Custom CSS for better UI
st.markdown("""
    <style>
        .big-font {
            font-size:30px !important;
        }
        .sidebar .sidebar-content {
            background-color: #f4f4f4;
        }
        .block-container {
            padding-top: 2rem;
            padding-bottom: 2rem;
        }
        .stButton>button {
            background-color: #4CAF50;
            color: white;
            font-size: 16px;
            padding: 12px;
            border-radius: 10px;
        }
        .stButton>button:hover {
            background-color: #45a049;
        }
    </style>
""", unsafe_allow_html=True)

# App title and configuration
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("🚗 Car Price Prediction App")

# Load pre-trained model and scaler with error handling
try:
    model = joblib.load('lasso_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError as e:
    st.error("🚨 Error: Model or Scaler file not found. Please ensure the files 'lasso_model.pkl' and 'scaler.pkl' are present.")
    st.stop()  # Stop further execution if files are missing

# Function to predict car price
def predict_car_price(features):
    features_scaled = scaler.transform([features])
    predicted_price = model.predict(features_scaled)[0]
    return predicted_price

# Sidebar for input features with icons and better layout
st.sidebar.header("🛠️ Enter Car Details")

present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, format="%.2f")
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=100)
year = st.sidebar.number_input("Year of Purchase", min_value=2000, max_value=2024, step=1)
fuel_type = st.sidebar.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"], help="Select the fuel type of the car.")
seller_type = st.sidebar.selectbox("Seller Type", options=["Dealer", "Individual"], help="Choose if the seller is a Dealer or Individual.")
transmission = st.sidebar.selectbox("Transmission", options=["Manual", "Automatic"], help="Select the type of transmission.")
owners = st.sidebar.number_input("Number of Previous Owners", min_value=0, step=1)

# Encoding categorical features
fuel_type_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_type_mapping = {"Dealer": 0, "Individual": 1}
transmission_mapping = {"Manual": 0, "Automatic": 1}

fuel_type_encoded = fuel_type_mapping[fuel_type]
seller_type_encoded = seller_type_mapping[seller_type]
transmission_encoded = transmission_mapping[transmission]

# Prepare features for prediction
features = [
    present_price,
    kms_driven,
    2024 - year,  # Age of the car
    fuel_type_encoded,
    seller_type_encoded,
    transmission_encoded,
    owners
]

# Prediction button with modern design
if st.sidebar.button("🔮 Predict Price"):
    predicted_price = predict_car_price(features)
    st.subheader("💡 Predicted Selling Price")
    st.markdown(f"### ₹ {predicted_price:,.2f} Lakhs")  # Display price in a readable format

# Section to showcase the dataset insights
st.sidebar.header("📊 Data Insights")
st.write(
    """
    This app predicts the selling price of used cars based on various features:
    - Present Price (in Lakhs)
    - Kilometers Driven
    - Year of Purchase
    - Fuel Type
    - Seller Type
    - Transmission Type
    - Number of Previous Owners
    
    The model used is a **Lasso Regression** trained on a dataset of car sales.
    """
)

# Show dataset insights with charts (optional)
if st.sidebar.checkbox("Show Data Insights"):
    st.subheader("📈 Dataset Insights")
    uploaded_file = st.file_uploader("Upload your car sales dataset for analysis", type=["csv"])
    
    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("### Dataset Preview:")
        st.dataframe(data.head())
        
        st.write("### Fuel Type Distribution:")
        fuel_type_count = data['Fuel_Type'].value_counts()
        st.bar_chart(fuel_type_count)
        
        st.write("### Selling Price Distribution:")
        st.histogram(data['Selling_Price'], bins=30)

# Developer Section with modern design
if st.sidebar.checkbox("👨‍💻 About the Developer"):
    st.subheader("About the Developer")
    st.write("Name: **Gurjap Singh**")
    st.write("Age: **17**")
    st.write("Passionate about **AI** and **Machine Learning**.")
    st.write("[LinkedIn Profile](https://www.linkedin.com/in/gurjap-singh-46696332a/)")

# Footer section for additional info
st.markdown("""
    ---
    🛠️ Built with ❤️ using Streamlit and Scikit-learn. For more details, contact the developer on LinkedIn.
""")
