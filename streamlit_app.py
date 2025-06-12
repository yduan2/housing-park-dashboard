import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import seaborn as sns

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
    return df_encoded

df = load_data()
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Train model ---
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
st.subheader("📌 Top 20 Feature Importances (Random Forest)")
importances = model.feature_importances_
indices = np.argsort(importances)[-20:]
top_features = X.columns[indices]
top_values = importances[indices]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(top_features, top_values, color='skyblue')
ax.set_xlabel("Importance")
ax.set_title("Top 20 Important Features")
st.pyplot(fig)

# --- Correlation Heatmap ---
st.subheader("🧮 Correlation Heatmap (Raw Features)")
with st.expander("Show Correlation Matrix"):
    corr = df.corr(numeric_only=True)
    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax_corr)
    st.pyplot(fig_corr)
