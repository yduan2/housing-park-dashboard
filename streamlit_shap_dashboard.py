import streamlit as st
import pandas as pd
import shap
import matplotlib.pyplot as plt
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title("🏡 Housing + Park Model Explainability Dashboard")

@st.cache_data
def load_data():
    df = pd.read_csv("RealEstate_Georgia.csv")
    df['age'] = 2025 - df['yearBuilt']
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

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
r2 = r2_score(y_test, y_pred)

st.sidebar.subheader("📈 Model Performance")
st.sidebar.metric("RMSE", f"${rmse:,.0f}")
st.sidebar.metric("R²", f"{r2:.2f}")

st.subheader("🔍 SHAP Summary Plot")
X_sample = X.sample(1000, random_state=42)
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)

fig_bar = shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)

st.subheader("🎯 SHAP Feature Impact (Direction + Spread)")
fig_dot = shap.summary_plot(shap_values, X_sample, show=False)
st.pyplot(bbox_inches='tight', dpi=300, clear_figure=True)

st.subheader("🔬 Force Plot for a Single Prediction")
shap.initjs()
row_idx = st.slider("Select a sample index", 0, X_sample.shape[0] - 1, 0)
force_html = shap.force_plot(
    explainer.expected_value, shap_values[row_idx], X_sample.iloc[row_idx], matplotlib=False
)
st.components.v1.html(shap.save_html("force_plot.html", force_html), height=300)
