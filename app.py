import streamlit as st
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Configure page
st.set_page_config(page_title="Car Price Prediction", page_icon="🚗", layout="wide")

# App title
st.title("🚗 Car Price Prediction App")

# Sidebar menu at the top
st.sidebar.markdown(
    """
    <style>
    .menu-title {
        font-size: 18px;
        color: #4CAF50;
        font-weight: bold;
        margin-bottom: 10px;
    }
    .menu-item {
        padding: 10px 5px;
        font-size: 14px;
        border-radius: 5px;
        cursor: pointer;
        color: #4CAF50;
        background-color: #f9f9f9;
        margin-bottom: 5px;
    }
    .menu-item:hover {
        background-color: #4CAF50;
        color: white;
    }
    .sidebar-section {
        margin-bottom: 20px;
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

st.sidebar.markdown('<div class="menu-title">Menu</div>', unsafe_allow_html=True)
menu_selection = st.sidebar.radio(
    "",
    ["🏠 Home", "📊 Data Insights", "📖 About", "👨‍💻 Developer"],
    index=0,
    label_visibility="collapsed",
)

# Sidebar inputs
st.sidebar.markdown('<div class="menu-title">Enter Car Details</div>', unsafe_allow_html=True)

present_price = st.sidebar.number_input("Present Price (in lakhs)", min_value=0.0, step=0.1)
kms_driven = st.sidebar.number_input("Kilometers Driven", min_value=0, step=100)
year = st.sidebar.number_input("Year of Purchase", min_value=2000, max_value=2023, step=1)
fuel_type = st.sidebar.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
seller_type = st.sidebar.selectbox("Seller Type", ["Dealer", "Individual"])
transmission = st.sidebar.selectbox("Transmission", ["Manual", "Automatic"])
owners = st.sidebar.number_input("Number of Previous Owners", min_value=0, step=1)

# Function to load model and scaler
def load_model():
    try:
        model = joblib.load("lasso_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("🚨 Error: Model or Scaler file not found. Ensure 'lasso_model.pkl' and 'scaler.pkl' are in the same directory.")
        st.stop()

# Predict button
if st.sidebar.button("Predict Price"):
    model, scaler = load_model()

    # Prepare input for prediction
    features = np.array([
        present_price,
        kms_driven,
        2024 - year,
        {"Petrol": 0, "Diesel": 1, "CNG": 2}[fuel_type],
        {"Dealer": 0, "Individual": 1}[seller_type],
        {"Manual": 0, "Automatic": 1}[transmission],
        owners,
    ]).reshape(1, -1)

    # Scale the features
    features_scaled = scaler.transform(features)

    # Predict price and handle negatives
    predicted_price = max(model.predict(features_scaled)[0], 0)

    # Format price with commas
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

# Additional menu content
if menu_selection == "📊 Data Insights":
    st.header("📊 Data Insights")
    uploaded_file = st.file_uploader("Upload your dataset for analysis", type=["csv"])

    if uploaded_file:
        import pandas as pd
        import seaborn as sns
        import matplotlib.pyplot as plt

        data = pd.read_csv(uploaded_file)
        st.write("Dataset Preview:")
        st.dataframe(data.head())

        st.write("🚗 Fuel Type Distribution:")
        plt.figure(figsize=(8, 6))
        sns.countplot(data=data, x="Fuel_Type", palette="coolwarm")
        st.pyplot(plt)

        st.write("💲 Selling Price Distribution:")
        plt.figure(figsize=(8, 6))
        sns.histplot(data["Selling_Price"], kde=True, bins=30, color="green")
        st.pyplot(plt)

elif menu_selection == "📖 About":
    st.header("📖 About the App")
    st.write("This application predicts the selling price of used cars based on various features.")

elif menu_selection == "👨‍💻 Developer":
    st.header("👨‍💻 About the Developer")
    st.write("Name: Gurjap Singh")
    st.write("Age: 17")
    st.write("Enthusiast in AI and Machine Learning.")
    st.write("[LinkedIn Profile](https://www.linkedin.com/in/gurjap-singh-46696332a/)")
