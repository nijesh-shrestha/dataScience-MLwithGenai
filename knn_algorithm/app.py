import streamlit as st
import pandas as pd
import joblib

label_encoder = joblib.load("label_encoder.joblib")
model = joblib.load("knn_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_names = joblib.load("feature_names.joblib")

st.title("Breast Cancer Prediction")

user_input = {}

for feature in feature_names:
    user_input[feature] = st.number_input(feature, value=0.0)

if st.button("Predict"):
    df = pd.DataFrame([user_input])

    scaled = scaler.transform(df)
    prediction = model.predict(scaled)
    label = label_encoder.inverse_transform(prediction)[0]

    if label == "M":
        st.error("Prediction: Malignant | Cancerous")
    else:
        st.success("Prediction: Benign | Not Cancerous")