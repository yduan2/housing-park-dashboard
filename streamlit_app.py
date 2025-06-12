import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="GA Housing + Walkability Dashboard", layout="wide")
st.title("🏠 Georgia Housing, Neighborhood, and Walk Score Dashboard")

# --- Load Housing Data ---
@st.cache_data
def load_housing_data():
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

df_encoded, df_raw = load_housing_data()
X = df_encoded.drop(columns=['price'])
y = df_encoded['price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# --- Sidebar: Performance ---
st.sidebar.header("📈 Model Performance")
st.sidebar.metric("RMSE", f"${rmse:,.0f}")
st.sidebar.metric("R²", f"{r2:.2f}")

# --- Section 1: Price Prediction Map ---
st.subheader("Predicted Home Prices Map")
df_map = X_test.copy()
df_map['ActualPrice'] = y_test.values
df_map['PredictedPrice'] = y_pred

fig_map = px.scatter_mapbox(
    df_map[(df_map['PredictedPrice'] > 10000) & (df_map['PredictedPrice'] < 2e6)],
    lat="latitude",
    lon="longitude",
    color="PredictedPrice",
    size="PredictedPrice",
    hover_data=["ActualPrice"],
    mapbox_style="carto-positron",
    zoom=6,
    color_continuous_scale="viridis",
    height=600
)
st.plotly_chart(fig_map, use_container_width=True)

# --- Section 2: Neighborhood Trends ---
@st.cache_data
def load_trend_data():
    df = pd.read_csv("GANeighborhood.csv")
    date_columns = df.columns[7:]
    df_long = df.melt(
        id_vars=["RegionName", "City", "CountyName", "Metro"],
        value_vars=date_columns,
        var_name="Date",
        value_name="HomeValue"
    )
    df_long["Date"] = pd.to_datetime(df_long["Date"], errors='coerce')
    df_long.dropna(subset=["HomeValue", "Date"], inplace=True)
    return df_long

df_trend = load_trend_data()

st.sidebar.header("Filter Trends")
cities = st.sidebar.multiselect("Select Cities", sorted(df_trend["City"].dropna().unique()), default=["Atlanta", "Sandy Springs"])
counties = st.sidebar.multiselect("Select Counties", sorted(df_trend["CountyName"].dropna().unique()))

trend_filtered = df_trend.copy()
if cities:
    trend_filtered = trend_filtered[trend_filtered["City"].isin(cities)]
if counties:
    trend_filtered = trend_filtered[trend_filtered["CountyName"].isin(counties)]

st.subheader("Home Value Trends by Neighborhood")
fig_trend = px.line(
    trend_filtered,
    x="Date",
    y="HomeValue",
    color="City",
    line_group="RegionName",
    hover_name="RegionName",
    labels={"HomeValue": "Home Value ($)", "Date": "Date"},
    height=600
)
st.plotly_chart(fig_trend, use_container_width=True)

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("Final.csv", encoding='ISO-8859-1')

# Define relevant columns
cols = [
    'price', 'WalkScore_FinalWeighted', 'householdtotals_TOTHH_CY',
    'HistoricalPopulation_TOTPOP_CY', 'householdincome_MEDHINC_CY',
    'householdincome_PCI_CY', 'householdincome_GINI_CY',
    'educationalattainment_BACHDEG_CY', 'crime_CRMCYTOTC',
    'Total_Park_Need_Score_Ranking', 'Total_Park_Need_Score',
    'Park_Conditions_and_Park_Level_Funding_Score', 'Community_Need_Score',
    'Level_of_Service_Score', 'Community_Perception_Score_Average_',
    'Maintenance_Funding_Score'
]

# Data cleaning
df_model = df[cols].apply(pd.to_numeric, errors='coerce').dropna()

# Define X and y
X = df_model.drop(columns='price')
y = df_model['price']

# OLS Regression
X_const = sm.add_constant(X)
model = sm.OLS(y, X_const).fit()

# Ridge Regression
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
ridge = Ridge(alpha=1.0)
ridge.fit(X_scaled, y)

# Random Forest
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X, y)

# Combine results
comparison_df = pd.DataFrame({
    'OLS_Coefficient': model.params[1:].values,
    'Ridge_Coefficient': ridge.coef_,
    'RandomForest_Importance': rf.feature_importances_
}, index=X.columns)

# Streamlit App
st.title("Housing Price Model: Walkability, Socioeconomics, and Parks")

st.subheader("OLS vs Ridge Coefficient Comparison")
fig, ax = plt.subplots(figsize=(12, 6))
bar_width = 0.35
index = range(len(comparison_df))
ax.bar(index, comparison_df['OLS_Coefficient'], bar_width, label='OLS Coefficient')
ax.bar([i + bar_width for i in index], comparison_df['Ridge_Coefficient'], bar_width, label='Ridge Coefficient')
ax.set_xticks([i + bar_width / 2 for i in index])
ax.set_xticklabels(comparison_df.index, rotation=90)
ax.set_ylabel("Coefficient Value")
ax.set_title("OLS vs Ridge Coefficients")
ax.legend()
st.pyplot(fig)

st.subheader("Random Forest Feature Importance")
fig2, ax2 = plt.subplots(figsize=(10, 6))
sorted_idx = rf.feature_importances_.argsort()
ax2.barh(range(len(sorted_idx)), rf.feature_importances_[sorted_idx], align='center')
ax2.set_yticks(range(len(sorted_idx)))
ax2.set_yticklabels([X.columns[i] for i in sorted_idx])
ax2.set_xlabel("Importance Score")
ax2.set_title("Feature Importance (Random Forest)")
st.pyplot(fig2)

st.subheader("Model Summary (OLS)")
st.text(model.summary())

import streamlit as st
from PIL import Image

st.subheader("Actual vs. Predicted Housing Prices")
result_img = Image.open("result.png")
st.image(result_img, caption="Model Prediction Accuracy (Random Forest)", use_column_width=True)

st.subheader("Walk Score and Housing Price Distribution in Atlanta")
map_img = Image.open("correlationmap.jpg")
st.image(map_img, caption="Hex-based Map of WalkScore and House Prices", use_column_width=True)


