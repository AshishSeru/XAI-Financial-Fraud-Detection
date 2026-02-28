# This file will now exclusively handle dashboard-specific preprocessing changes.

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import joblib

def preprocess_dashboard_data(input_data):
    """
    Preprocesses the input data specifically for the dashboard pipeline.

    Args:
        input_data (pd.DataFrame): The raw input data from the dashboard.

    Returns:
        pd.DataFrame: The preprocessed data ready for anomaly detection and classification.
    """
    print("[INFO] Preprocessing dashboard data...")

    # Drop irrelevant columns if they exist
    drop_cols = ['TransactionID', 'AccountID']
    input_data = input_data.drop(columns=drop_cols, errors='ignore')

    # Handle date columns
    date_cols = ['TransactionDate', 'PreviousTransactionDate']
    for col in date_cols:
        if col in input_data.columns:
            input_data[col] = pd.to_datetime(input_data[col], errors='coerce')

    # Calculate time difference in seconds between transactions
    if 'TransactionDate' in input_data.columns and 'PreviousTransactionDate' in input_data.columns:
        input_data['TimeDifference'] = (input_data['TransactionDate'] - input_data['PreviousTransactionDate']).dt.total_seconds()

    # Load pre-fitted transformers
    numeric_imputer = joblib.load('models/numeric_imputer.pkl')
    numeric_scaler = joblib.load('models/numeric_scaler.pkl')

    # Impute and scale numeric columns
    numeric_cols = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'AccountBalance', 'TimeDifference']
    input_data[numeric_cols] = numeric_imputer.transform(input_data[numeric_cols])
    input_data[numeric_cols] = numeric_scaler.transform(input_data[numeric_cols])

    # Encode categorical features using pre-fitted encoders
    categorical_cols = ['TransactionType', 'Location', 'DeviceID', 'IP Address', 'MerchantID', 'Channel', 'CustomerOccupation']
    for col in categorical_cols:
        if col in input_data.columns:
            encoder = joblib.load(f'models/{col}_encoder.pkl')
            input_data[col] = encoder.transform(input_data[col].astype(str))

    print("[INFO] Dashboard data preprocessing complete.")
    return input_data