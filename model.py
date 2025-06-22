# model.py

import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("used_cars_data.csv")

# Drop missing values
df = df.dropna()

# Clean engine column (e.g., "1248 CC" → 1248)
df['Engine'] = df['Engine'].astype(str).str.extract('(\d+)').astype(float)

# Encode categorical variables
label_encoder = LabelEncoder()
df['car_name_encoded'] = label_encoder.fit_transform(df['Name'])

fuel_map = {"Petrol": 0, "Diesel": 1, "CNG": 2, "Electric": 3, "LPG": 4}
trans_map = {"Manual": 0, "Automatic": 1}
owner_map = {"First Owner": 0, "Second Owner": 1, "Third Owner": 2, 
             "Fourth & Above Owner": 3, "Test Drive Car": 4}

df['fuel_encoded'] = df['Fuel_Type'].map(fuel_map).fillna(0)
df['trans_encoded'] = df['Transmission'].map(trans_map).fillna(0)
df['owner_encoded'] = df['Owner_Type'].map(owner_map).fillna(0)

# Feature selection
features = ['car_name_encoded', 'Year', 'Kilometers_Driven', 
            'fuel_encoded', 'trans_encoded', 'owner_encoded', 'Engine']
X = df[features]
y = df['Price']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a small model (to keep file size < 25MB)
model = RandomForestRegressor(n_estimators=30, max_depth=10, random_state=42)
model.fit(X_train, y_train)

# Save model and label encoder
joblib.dump(model, "car_price_model_small.pkl")
joblib.dump(label_encoder, "label_encoder_small.pkl")

print("✅ Model and encoder saved as car_price_model_small.pkl and label_encoder_small.pkl")
