import streamlit as st
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Set page config with an icon and a title
st.set_page_config(page_title="Car Price Prediction", layout="wide", page_icon="🚗")

# App title with an emoji
st.title("🚗 Car Price Prediction App")

# Provide a brief introduction
st.write("""
Welcome to the Car Price Prediction App! 
Enter the details of your car, and we'll predict the selling price for you. 
Just fill in the details and click 'Predict Price'! 💡
""")

# Sidebar with input fields
st.sidebar.header("Enter Car Details")

# Input fields for car features
present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1, value=18.0)
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=100, value=80000)
year = st.sidebar.number_input("Year of Purchase", min_value=2000, step=1, value=2019)
fuel_type = st.sidebar.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", options=["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", options=["Manual", "Automatic"])
owners = st.sidebar.number_input("Number of Previous Owners", min_value=0, step=1, value=1)

# Mapping categorical input values to numerical
fuel_type_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
seller_type_mapping = {"Dealer": 0, "Individual": 1}
transmission_mapping = {"Manual": 0, "Automatic": 1}

fuel_type_encoded = fuel_type_mapping[fuel_type]
seller_type_encoded = seller_type_mapping[seller_type]
transmission_encoded = transmission_mapping[transmission]

# Button to trigger the prediction
if st.sidebar.button("Predict Price"):
    st.sidebar.write("Predicting...")

    # Load model and scaler
    def load_model():
        try:
            model = joblib.load('lasso_model.pkl')
            scaler = joblib.load('scaler.pkl')
            return model, scaler
        except FileNotFoundError:
            st.error("Model or Scaler file not found. Ensure 'lasso_model.pkl' and 'scaler.pkl' are present.")
            return None, None

    model, scaler = load_model()

    if model and scaler:
        # Prepare input data for prediction
        car_features = np.array([
            present_price,
            kms_driven,
            2024 - year,  # Calculate car age
            fuel_type_encoded,
            seller_type_encoded,
            transmission_encoded,
            owners
        ]).reshape(1, -1)

        # Scale the input data
        car_features_scaled = scaler.transform(car_features)

        # Make prediction
        predicted_price = model.predict(car_features_scaled)[0]

        # Display the predicted price
        st.subheader("Predicted Selling Price:")
        st.write(f"₹ {predicted_price:.2f} Lakhs")

# Option to show data insights
if st.sidebar.checkbox("Show Data Insights"):
    st.subheader("Data Insights")

    uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

    if uploaded_file:
        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        st.write("Fuel Type Distribution:")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x="Fuel_Type", palette="viridis")
        st.pyplot(plt)

        st.write("Selling Price Distribution:")
        plt.figure(figsize=(8, 6))
        sns.histplot(data["Selling_Price"], kde=True, bins=30, color="skyblue")
        st.pyplot(plt)
