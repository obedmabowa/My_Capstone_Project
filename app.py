import streamlit as st
import pandas as pd
import plotly.express as px
from model import train_model

st.set_page_config(page_title="Avocado Price Predictor", layout="wide")

# -------------------------------
# AUTH
# -------------------------------
def login():
    st.title("🥑 Avocado App Login")

    user = st.text_input("Username")
    pwd = st.text_input("Password", type="password")

    if st.button("Login"):
        if user == "admin" and pwd == "admin":
            st.session_state["auth"] = True
        else:
            st.error("Invalid credentials")

if "auth" not in st.session_state:
    login()
    st.stop()

# -------------------------------
# LOAD DATA + MODEL
# -------------------------------
@st.cache_data
def load_data():
    return pd.read_csv("Avocado_Prices_Data.csv")

@st.cache_resource
def load_model():
    return train_model()

df = load_data()
model, features = load_model()

df["Date"] = pd.to_datetime(df["Date"])
df["Year"] = df["Date"].dt.year

# -------------------------------
# SIDEBAR
# -------------------------------
page = st.sidebar.radio("Navigation", ["EDA", "Predictor"])

# -------------------------------
# EDA PAGE
# -------------------------------
if page == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.dataframe(df.head())

    fig1 = px.line(df, x="Date", y="AveragePrice", title="Average Price Over Time")
    st.plotly_chart(fig1, use_container_width=True)

    fig2 = px.box(df, x="region", y="AveragePrice", title="Price by Region")
    st.plotly_chart(fig2, use_container_width=True)

# -------------------------------
# PREDICTOR PAGE
# -------------------------------
if page == "Predictor":
    st.title("🤖 Avocado Price Predictor")

    region = st.selectbox("Select Geography", df["region"].unique())
    year = st.slider("Select Year", int(df["Year"].min()), int(df["Year"].max()))

    volume = st.number_input("Total Volume", value=100000.0)
    bags = st.number_input("Total Bags", value=50000.0)

    if st.button("Predict Price"):
        input_df = pd.DataFrame(columns=features)
        input_df.loc[0] = 0

        input_df["Year"] = year
        input_df["Total Volume"] = volume
        input_df["Total Bags"] = bags

        region_col = f"region_{region}"
        if region_col in input_df.columns:
            input_df[region_col] = 1

        prediction = model.predict(input_df)[0]

        st.success(f"🥑 Predicted Price: **${prediction:.2f}**")
