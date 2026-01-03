import streamlit as st
import pandas as pd
import joblib

# Page configuration
st.set_page_config(page_title="Intrusion Detection System", layout="wide")

# Title
st.title("Intrusion Detection System")
st.write(
    "This application uses a trained Random Forest model to detect "
    "network intrusions from raw traffic data."
)

# Load the trained pipeline (preprocessing + model)
pipeline = joblib.load("rf_pipeline.pkl")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a CSV file containing raw network traffic data",
    type=["csv"]
)

if uploaded_file is not None:
    # Read CSV
    data = pd.read_csv(uploaded_file)

    # Drop unnecessary columns if present
    data = data.drop(columns=['Unnamed: 0'], errors='ignore')

    st.subheader("Preview of Uploaded Data")
    st.dataframe(data.head())

    if st.button("Run Intrusion Detection"):
        # Make predictions
        predictions = pipeline.predict(data)

        # Add predictions to dataframe
        data["Prediction"] = predictions

        st.success("Prediction completed successfully!")

        st.subheader("Prediction Results")
        st.dataframe(data.head())

        # Optional: show class distribution
        st.subheader("Prediction Distribution")
        st.bar_chart(data["Prediction"].value_counts())
