import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ------------------- Load Model, Encoder, and Dataset -------------------
model = joblib.load("car_price_model_small.pkl")
label_encoder = joblib.load("label_encoder_small.pkl")
df = pd.read_csv("used_cars_data.csv")

# ------------------- Clean Dataset -------------------
df_clean = df.dropna().copy()
df_clean['Engine'] = df_clean['Engine'].astype(str).str.extract('(\d+)').astype(float)

# ------------------- Encoding Maps -------------------
fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "LPG": 4}
trans_map = {"Manual": 0, "Automatic": 1}
owner_map = {
    "First Owner": 0, "Second Owner": 1, "Third Owner": 2,
    "Fourth & Above Owner": 3, "Test Drive Car": 4
}

# ------------------- Page Configuration -------------------
st.set_page_config(page_title="Used Car Price Predictor", layout="wide")

# ------------------- Sidebar Navigation -------------------
st.sidebar.title("🔍 Navigation")
page = st.sidebar.radio("Go to", ["Home", "Dataset", "Predict"])

# ------------------- HOME PAGE -------------------
if page == "Home":
    st.title("🚗 Used Car Price Predictor")
    st.markdown("""
        Welcome to the **Used Car Price Predictor** app! This app uses machine learning to estimate the resale price of used cars based on:
        - Car model
        - Year of manufacture
        - Kilometers driven
        - Fuel type
        - Transmission
        - Ownership history
        - Engine capacity

        👉 Use the sidebar to explore the dataset or predict prices.
    """)


    st.subheader("📉 Fuel Type Distribution")
    fuel_counts = df_clean['Fuel_Type'].value_counts()
    st.bar_chart(fuel_counts)

    

    st.subheader("📌 Model Details")
    st.markdown("""
        - **Model Used**: Random Forest Regressor *(example, may vary)*
        - **Input Features**: Encoded categorical + numerical
        - **Typical Accuracy**: ~85% (on test set)
        - **Training Data**: Real-world used car records
    """)

# ------------------- DATASET PAGE -------------------
elif page == "Dataset":
    st.title("📊 Dataset Preview")
    st.markdown("Below is the dataset used to train the model:")
    st.dataframe(df_clean.reset_index(drop=True))

# ------------------- PREDICT PAGE -------------------
elif page == "Predict":
    st.title("🤖 Predict Used Car Price")

    available_cars = sorted(label_encoder.classes_)

    car_name = st.selectbox("Select Car Model", available_cars)
    year = st.number_input("Year", min_value=1990, max_value=2025, value=2015)
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    fuel = st.selectbox("Fuel Type", list(fuel_map.keys()))
    trans = st.selectbox("Transmission", list(trans_map.keys()))
    owner = st.selectbox("Owner Type", list(owner_map.keys()))
    engine = st.number_input("Engine (in CC)", min_value=500, max_value=10000, value=1200)

    if st.button("Predict Price"):
        try:
            name_encoded = label_encoder.transform([car_name])[0]
            fuel_encoded = fuel_map[fuel]
            trans_encoded = trans_map[trans]
            owner_encoded = owner_map[owner]

            features = np.array([[name_encoded, year, km_driven, fuel_encoded, trans_encoded, owner_encoded, engine]])
            prediction = model.predict(features)[0]
            st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} lakhs")

        except Exception as e:
            st.error(f"Prediction failed due to: {e}")
