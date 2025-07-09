# app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib

from src.preprocess import preprocess_data

st.set_page_config(page_title="🏦 Loan Predictor", layout="centered")
st.title("🏦 Loan Eligibility Predictor")
st.markdown("Enter applicant details to check loan approval status")

# Load model + encoders
model = joblib.load('models/loan_model.pkl')
encoders = joblib.load('models/label_encoders.pkl')

# Input fields
gender = st.selectbox("Gender", ['Male', 'Female'])
married = st.selectbox("Married", ['Yes', 'No'])
dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
self_employed = st.selectbox("Self Employed", ['Yes', 'No'])
applicant_income = st.number_input("Applicant Income", min_value=0)
coapplicant_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
loan_term = st.selectbox("Loan Amount Term", [360.0, 120.0, 180.0, 300.0, 240.0])
credit_history = st.selectbox("Credit History", [1.0, 0.0])
property_area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

# Predict
if st.button("Predict"):
    input_df = pd.DataFrame([{
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': applicant_income,
        'CoapplicantIncome': coapplicant_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area,
    }])

    X, _ = preprocess_data(input_df, encoders=encoders, is_train=False)
    prediction = model.predict(X)

    if prediction[0] == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
