import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

st.set_page_config(page_title="GA Housing + Park Dashboard", layout="wide")
st.title("🏠 Georgia Housing Price Prediction & Neighborhood Trends")

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

# --- Scatter Map of Predictions ---
st.subheader("🗼️ Predicted Home Prices Map")
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

# --- Load Neighborhood Trend Data ---
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

# --- Sidebar Filters ---
st.sidebar.header("📍 Filter Neighborhood Trends")
cities = st.sidebar.multiselect("Select Cities", sorted(df_trend["City"].dropna().unique()), default=["Atlanta", "Sandy Springs"])
counties = st.sidebar.multiselect("Select Counties", sorted(df_trend["CountyName"].dropna().unique()))

trend_filtered = df_trend.copy()
if cities:
    trend_filtered = trend_filtered[trend_filtered["City"].isin(cities)]
if counties:
    trend_filtered = trend_filtered[trend_filtered["CountyName"].isin(counties)]

# --- Trend Plot ---
st.subheader("📊 Home Value Trends by Neighborhood")
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

# --- Walk Score Map ---
st.title("🌍 Atlanta Walk Score Explorer Dashboard")

@st.cache_data
def load_walkscore_data():
    df = pd.read_csv("walkscore.csv")
    df = df.dropna(subset=["POINT_X", "POINT_Y", "WalkScore_FinalWeighted"])
    df = df[df["WalkScore_FinalWeighted"] > 0]
    return df

walk_df = load_walkscore_data()

st.sidebar.header("Walk Score Filter")
min_score = int(walk_df["WalkScore_FinalWeighted"].min())
max_score = int(walk_df["WalkScore_FinalWeighted"].max())
score_range = st.sidebar.slider("Select Walk Score Range", min_value=min_score, max_value=max_score, value=(min_score, max_score))

filtered_df = walk_df[
    (walk_df["WalkScore_FinalWeighted"] >= score_range[0]) &
    (walk_df["WalkScore_FinalWeighted"] <= score_range[1])
]

st.subheader("🗼️ Walk Score Map (Filtered)")
fig_walk = px.scatter_mapbox(
    filtered_df,
    lat="POINT_Y",
    lon="POINT_X",
    color="WalkScore_FinalWeighted",
    size="WalkScore_FinalWeighted",
    size_max=12,
    zoom=10,
    mapbox_style="carto-positron",
    color_continuous_scale="Viridis",
    hover_data=["Sidewalks", "Intersections", "POI", "2024 Median Household Income"]
)
fig_walk.update_layout(margin={"r":0, "t":30, "l":0, "b":0}, height=650)
st.plotly_chart(fig_walk, use_container_width=True)

# Optional: Data Table
with st.expander("📊 View Walk Score Data Table"):
    st.dataframe(filtered_df)

st.markdown("---")
st.caption("Developed by Yan Duan | 2025")
