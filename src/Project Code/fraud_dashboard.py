import os
import sys

# Add the project root directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
from joblib import load
from datetime import datetime
from src.dashboard_preprocessing import preprocess_dashboard_data
from src.explainability import shap_explanation_classifier as get_shap_explanation
from src.classifier import predict_fraud

# Replace pickle with joblib for loading the model
model_path = "models/fraud_classifier.pkl"
model = load(model_path)

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 Real-Time Fraud Detection & Explainability")

st.markdown("Fill in the transaction details to check the fraud probability and explanation.")

# Transaction input form
with st.form("txn_form"):
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction Amount", min_value=0.0)
        txn_type = st.selectbox("Transaction Type", ["Debit", "Credit"])
        location = st.text_input("Location")
        device_id = st.text_input("Device ID")
        ip_address = st.text_input("IP Address")
        merchant_id = st.text_input("Merchant ID")
        channel = st.selectbox("Channel", ["ATM", "Online", "POS"])
    with col2:
        age = st.number_input("Customer Age", min_value=10, max_value=100)
        occupation = st.selectbox("Customer Occupation", ["Doctor", "Engineer", "Student", "Business", "Other"])
        duration = st.number_input("Transaction Duration (seconds)", min_value=1)
        login_attempts = st.number_input("Login Attempts", min_value=0)
        balance = st.number_input("Account Balance", min_value=0.0)
        txn_date = st.text_input("Transaction Date (DD-MM-YYYY HH:MM)", value="01-01-2023 12:00")
        prev_txn_date = st.text_input("Previous Transaction Date (DD-MM-YYYY HH:MM)", value="01-01-2023 10:00")

    submitted = st.form_submit_button("Detect Fraud")

if submitted:
    try:
        input_data = pd.DataFrame([{
            "TransactionAmount": amount,
            "TransactionType": txn_type,
            "Location": location,
            "DeviceID": device_id,
            "IP Address": ip_address,
            "MerchantID": merchant_id,
            "Channel": channel,
            "CustomerAge": age,
            "CustomerOccupation": occupation,
            "TransactionDuration": duration,
            "LoginAttempts": login_attempts,
            "AccountBalance": balance,
            "TransactionDate": txn_date,
            "PreviousTransactionDate": prev_txn_date
        }])

        # Debugging: Verify if the loaded model supports predict_proba
        if hasattr(model, 'predict_proba'):
            st.write("[DEBUG] Model supports predict_proba.")
        else:
            st.write("[DEBUG] Model does NOT support predict_proba.")

        # Check if the loaded model is compatible
        if not hasattr(model, 'predict_proba'):
            raise ValueError("The loaded model does not support 'predict_proba'. Ensure 'fraud_classifier.pkl' contains a valid scikit-learn classifier.")

        # Debugging: Log input data before preprocessing
        st.write("[DEBUG] Input data before preprocessing:", input_data)

        # Step 1: Preprocess the input data
        st.write("[DEBUG] Preprocessing input data...")
        processed = preprocess_dashboard_data(input_data)
        st.write("[DEBUG] Preprocessed data:", processed)

        # Ensure compatibility by dropping the TimeDifference column
        processed = processed.drop(columns=['TimeDifference'], errors='ignore')

        # Debugging: Log data after dropping TimeDifference
        st.write("[DEBUG] Data after dropping TimeDifference:", processed)

        # Ensure compatibility with the models by dropping unnecessary columns
        processed_for_anomaly = processed.drop(columns=['TransactionID', 'AccountID', 'TransactionDate', 'PreviousTransactionDate'], errors='ignore')

        # Step 2: Add anomaly score using Isolation Forest
        st.write("[DEBUG] Adding anomaly score...")
        isolation_model_path = "models/isolation_forest.pkl"
        isolation_model = load(isolation_model_path)
        processed['AnomalyScore'] = isolation_model.decision_function(processed_for_anomaly)
        st.write("[DEBUG] Data with anomaly score:", processed)

        # Ensure compatibility by dropping date columns
        processed_for_classifier = processed.drop(columns=['TransactionDate', 'PreviousTransactionDate'], errors='ignore')

        # Step 3: Predict fraud using the classifier
        st.write("[DEBUG] Predicting fraud...")
        prob, pred = predict_fraud(model, processed_for_classifier)
        st.write("[DEBUG] Prediction result:", {"Probability": prob, "Prediction": pred})

        # Display prediction results
        st.markdown(f"### 🧮 Prediction: {'🟥 FRAUD' if pred else '🟩 Not Fraud'}")
        st.markdown(f"**Probability of Fraud:** {prob:.2%}")

        # Step 4: Explain prediction using SHAP
        st.write("[DEBUG] Generating SHAP explanations...")
        explanation_df = get_shap_explanation(processed_for_classifier, model=model)
        st.subheader("🔍 Reason for Prediction")
        st.dataframe(explanation_df)

    except Exception as e:
        st.error(f"Error: {e}")
