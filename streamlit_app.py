import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Set Streamlit page layout
st.set_page_config(page_title="Housing + Park Model", layout="wide")
st.title("🏡 Housing + Park Model Dashboard")

# --- Load and preprocess data ---
@st.cache_data
def load_data():
    df = pd.read_csv("RealEstate_Georgia.csv")
    df["age"] = 2025 - df["yearBuilt"]

    categorical = ['city', 'county', 'homeType']
    numeric = [
        'livingArea', 'bedrooms', 'bathrooms', 'garageSpaces',
        'parking', 'hasGarage', 'pool', 'spa', 'isNewConstruction',
        'hasPetsAllowed', 'latitude', 'longitude', 'age'
    ]
    df_clean = df[numeric + categorical + ['price']].dropna()
    df_encoded = pd.get_dummies(df_clean, columns=categorical, drop_first=True)
    return df_encoded, df_clean

df_encoded, df_raw = load_data()
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train Random Forest model ---
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --- Sidebar Metrics ---
st.sidebar.header("📊 Model Performance")
st.sidebar.metric("RMSE", f"${rmse:,.0f}")
st.sidebar.metric("R²", f"{r2:.2f}")

# --- Feature Importance ---
st.subheader("🔎 Feature Importance (Top 20)")
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]
top_features = X.columns[indices]
top_values = importances[indices]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_features, top_values, color='cornflowerblue')
ax.set_xlabel("Importance")
ax.set_title("Top 20 Important Features")
st.pyplot(fig)

# --- Geospatial Visualization ---
st.subheader("🗺️ Predicted Housing Prices Map")

# Add lat/lon + predictions back into raw data
df_map = df_raw.copy()
df_map["PredictedPrice"] = model.predict(X)

# Filter out extreme outliers
df_map = df_map[(df_map["PredictedPrice"] > 10000) & (df_map["PredictedPrice"] < 2_000_000)]

fig_map = px.scatter_mapbox(
    df_map,
    lat="latitude",
    lon="longitude",
    color="PredictedPrice",
    size="PredictedPrice",
    size_max=15,
    zoom=7,
    mapbox_style="carto-positron",
    color_continuous_scale="viridis",
    hover_data=["price"]
)

fig_map.update_layout(height=600, margin={"r":0,"t":30,"l":0,"b":0})
st.plotly_chart(fig_map, use_container_width=True)
