import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

st.title("🏠 Housing Price Prediction Dashboard")

# Load and clean data
@st.cache_data
def load_data():
    df = pd.read_csv("RealEstate_Georgia.csv")
    df["age"] = 2025 - df["yearBuilt"]
    cols = ['livingArea', 'bedrooms', 'bathrooms', 'garageSpaces', 'latitude', 'longitude', 'age',
            'city', 'county', 'homeType', 'price']
    df = df[cols].dropna()
    df = pd.get_dummies(df, columns=['city', 'county', 'homeType'], drop_first=True)
    return df

df = load_data()
X = df.drop(columns=['price'])
y = df['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Show model metrics
st.write("### Model Performance")
st.write(f"**RMSE:** ${np.sqrt(mean_squared_error(y_test, y_pred)):.0f}")
st.write(f"**R² Score:** {r2_score(y_test, y_pred):.2f}")

# Show feature importance
st.write("### Feature Importance")
importances = model.feature_importances_
indices = np.argsort(importances)[-10:]
plt.figure(figsize=(6, 4))
plt.barh(X.columns[indices], importances[indices])
plt.xlabel("Importance")
plt.tight_layout()
st.pyplot(plt)

# Show correlation heatmap
st.write("### Correlation Heatmap")
corr = df.corr(numeric_only=True)
plt.figure(figsize=(6, 5))
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.tight_layout()
st.pyplot(plt)
