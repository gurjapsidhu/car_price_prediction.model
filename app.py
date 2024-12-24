import streamlit as st
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# App title
st.set_page_config(page_title="Car Price Prediction", layout="wide")
st.title("Car Price Prediction App")

# Menu bar with three menus
menu = st.sidebar.radio("Menu", ["Predict Car Price", "About App & Model", "About Developer"])

if menu == "Predict Car Price":
    st.header("Car Price Prediction")

    # Function to load the model
    def load_model():
        try:
            model = joblib.load('lasso_model.pkl')
            return model
        except FileNotFoundError:
            st.error("Model file not found. Please train the model first.")
            return None

    # Input fields for car features
    st.sidebar.header("Provide the following details:")

    present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
    kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=100)
    year = st.sidebar.number_input("Year of Purchase", min_value=2000, step=1)
    fuel_type = st.sidebar.selectbox("Fuel Type", options=["Petrol", "Diesel", "CNG"])
    seller_type = st.sidebar.selectbox("Seller Type", options=["Dealer", "Individual"])
    transmission = st.sidebar.selectbox("Transmission", options=["Manual", "Automatic"])
    owners = st.sidebar.number_input("Number of Previous Owners", min_value=0, step=1)

    # Encoding categorical inputs
    fuel_type_mapping = {"Petrol": 0, "Diesel": 1, "CNG": 2}
    seller_type_mapping = {"Dealer": 0, "Individual": 1}
    transmission_mapping = {"Manual": 0, "Automatic": 1}

    fuel_type_encoded = fuel_type_mapping[fuel_type]
    seller_type_encoded = seller_type_mapping[seller_type]
    transmission_encoded = transmission_mapping[transmission]

    # Button to predict price
    if st.sidebar.button("Predict Price"):
        model = load_model()
        if model:
            # Prepare input data for prediction
            car_features = np.array([
                present_price,
                kms_driven,
                2024 - year,  # Calculate the age of the car
                fuel_type_encoded,
                seller_type_encoded,
                transmission_encoded,
                owners
            ]).reshape(1, -1)

            # Predicting price
            predicted_price = model.predict(car_features)[0]

            # Convert predicted price to lakhs (if needed, divide by 100,000)
            predicted_price_in_lakhs = predicted_price / 100000

            # Display result with formatted currency in lakhs
            st.subheader("💰 Predicted Selling Price")
            st.write(f"**₹ {predicted_price_in_lakhs:,.2f} Lakh**")

    # Visualization option
    if st.sidebar.checkbox("Show Data Insights"):
        st.subheader("Visualization")
        uploaded_file = st.file_uploader("Upload your dataset for analysis", type=["csv"])

        if uploaded_file:
            data = pd.read_csv(uploaded_file)
            st.write("Dataset Preview:")
            st.dataframe(data.head())

            st.write("Fuel Type Distribution:")
            plt.figure(figsize=(8, 6))
            sns.countplot(data=data, x="Fuel_Type")
            st.pyplot(plt)

            st.write("Selling Price Distribution:")
            plt.figure(figsize=(8, 6))
            sns.histplot(data["Selling_Price"], kde=True, bins=30)
            st.pyplot(plt)

elif menu == "About App & Model":
    st.header("About the App")
    st.write("This application predicts the selling price of used cars based on various features including price, kilometers driven, year of purchase, fuel type, seller type, transmission, and the number of previous owners.")
    st.write("The model used for prediction is a Lasso Regression model trained on a dataset of car sales.")
    st.write("### Features of the App")
    st.write("- User-friendly input form for car details.")
    st.write("- Prediction of car selling prices.")
    st.write("- Insights and visualizations based on the dataset.")
    st.write("### Evaluation Techniques")
    st.write("The model evaluation is based on R² score, which measures the proportion of variance in the dependent variable that is predictable from the independent variables.")

elif menu == "About Developer":
    st.header("About the Developer")
    st.write("Name: Gurjap Singh")
    st.write("Age: 17")
    st.write("Enthusiast in AI and Machine Learning.")
    st.write("[LinkedIn Profile](https://www.linkedin.com/in/gurjap-singh-46696332a/)")
