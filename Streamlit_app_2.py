import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

# Load the trained model
model = joblib.load('random_forest_model.pkl')

# Define the preprocessor (assuming you used a similar preprocessor in training)
categorical_features = ['type', 'region']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ],
    remainder='passthrough'
)

# Create a pipeline with the preprocessor and model
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])

# Streamlit app
st.set_page_config(page_title="Avocado Price Predictor & EDA", layout="wide")
st.title("🥑 Avocado Price Prediction & EDA App")

# Sidebar navigation
st.sidebar.title("Navigation")
options = st.sidebar.radio("Choose an Option:", ['Predict Price', 'EDA'])

# Dataset for EDA
@st.cache_data
def load_data():
    return pd.read_csv("avocado.csv")  # Replace with your dataset file

# Option: Predict Price
if options == 'Predict Price':
    st.header("Price Prediction")
    st.write("""
    Provide the avocado features below to predict the price.
    """)

    # Sidebar for categorical inputs
    st.sidebar.header("Avocado Details")
    type_ = st.sidebar.selectbox("Type", ['conventional', 'organic'], help="Type of avocado (conventional or organic)")
    region = st.sidebar.selectbox("Region", ['Albany', 'Atlanta', 'BaltimoreWashington'], help="Region where the avocado is sold")

    # Main input area for numeric features
    st.subheader("Enter Avocado Features:")
    col1, col2, col3 = st.columns(3)

    with col1:
        date = st.date_input("Date", help="Date for which the price prediction is required")
        average_price = st.number_input("Average Price", min_value=0.0, step=0.01, help="Average price of the avocado")
        total_volume = st.number_input("Total Volume", min_value=0.0, step=0.01, help="Total volume of avocados sold")

    with col2:
        plu4046 = st.number_input("PLU4046", min_value=0.0, step=0.01, help="Volume of PLU4046 avocados sold")
        plu4225 = st.number_input("PLU4225", min_value=0.0, step=0.01, help="Volume of PLU4225 avocados sold")
        plu4770 = st.number_input("PLU4770", min_value=0.0, step=0.01, help="Volume of PLU4770 avocados sold")

    with col3:
        total_bags = st.number_input("Total Bags", min_value=0.0, step=0.01, help="Total number of bags sold")
        small_bags = st.number_input("Small Bags", min_value=0.0, step=0.01, help="Number of small bags sold")
        large_bags = st.number_input("Large Bags", min_value=0.0, step=0.01, help="Number of large bags sold")
        xlarge_bags = st.number_input("XLarge Bags", min_value=0.0, step=0.01, help="Number of extra-large bags sold")

    # Prepare data for prediction
    input_data = pd.DataFrame({
        'Date': [pd.to_datetime(date)],
        'AveragePrice': [average_price],
        'TotalVolume': [total_volume],
        'plu4046': [plu4046],
        'plu4225': [plu4225],
        'plu4770': [plu4770],
        'TotalBags': [total_bags],
        'SmallBags': [small_bags],
        'LargeBags': [large_bags],
        'XLargeBags': [xlarge_bags],
        'type': [type_],
        'region': [region]
    })

    # Predict and display results
    if st.button("Predict 🥑"):
        prediction = pipeline.predict(input_data.drop(columns='Date'))
        st.success(f"Predicted Avocado Price: **${prediction[0]:.2f}**")
        st.write("Prediction made based on the input features provided.")

# Option: EDA
elif options == 'EDA':
    st.header("Exploratory Data Analysis")
    st.write("Analyze the avocado dataset to gain insights.")

    # Load and display dataset
    data = load_data()
    if st.checkbox("Show Raw Data"):
        st.write(data.head())

    # Summary statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(data.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Price distribution by type
    st.subheader("Price Distribution by Type")
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.boxplot(data=data, x='type', y='AveragePrice', palette='pastel')
    st.pyplot(fig)

    # Volume trends over time
    st.subheader("Volume Trends Over Time")
    fig, ax = plt.subplots(figsize=(10, 5))
    data['Date'] = pd.to_datetime(data['Date'])
    volume_trends = data.groupby('Date')['TotalVolume'].mean().reset_index()
    sns.lineplot(data=volume_trends, x='Date', y='TotalVolume', ax=ax)
    st.pyplot(fig)

    # Filter by region
    st.subheader("Filter by Region")
    selected_region = st.selectbox("Select Region:", data['region'].unique())
    filtered_data = data[data['region'] == selected_region]
    st.write(filtered_data.head())
