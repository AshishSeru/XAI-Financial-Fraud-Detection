import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from src.config import DATA_PATH
import joblib

def load_data():
    print("[INFO] Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    return df

def preprocess_data(df):
    print("[INFO] Preprocessing dataset...")

    # Numeric features to process
    numeric_cols = ['TransactionAmount', 'CustomerAge', 'TransactionDuration', 'AccountBalance']

    # Handle date columns
    date_cols = ['TransactionDate', 'PreviousTransactionDate']
    for col in date_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors='coerce')

    # Calculate time difference in seconds between transactions
    if 'TransactionDate' in df.columns and 'PreviousTransactionDate' in df.columns:
        df['TimeDifference'] = (df['TransactionDate'] - df['PreviousTransactionDate']).dt.total_seconds()
        df = df.drop(columns=date_cols, errors='ignore')

    # Add TimeDifference to numeric columns
    numeric_cols.append('TimeDifference')

    # Save pre-fitted numeric transformers
    numeric_imputer = SimpleImputer(strategy='mean')
    df[numeric_cols] = numeric_imputer.fit_transform(df[numeric_cols])
    joblib.dump(numeric_imputer, 'models/numeric_imputer.pkl')

    numeric_scaler = StandardScaler()
    df[numeric_cols] = numeric_scaler.fit_transform(df[numeric_cols])
    joblib.dump(numeric_scaler, 'models/numeric_scaler.pkl')

    # Drop the TimeDifference column if it exists
    df = df.drop(columns=['TimeDifference'], errors='ignore')

    # Save pre-fitted encoders for categorical features
    categorical_cols = ['TransactionType', 'Location', 'DeviceID', 'IP Address', 'MerchantID', 'Channel', 'CustomerOccupation']
    for col in categorical_cols:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col].astype(str))
        joblib.dump(encoder, f'models/{col}_encoder.pkl')

    print("[INFO] Preprocessing complete.")
    return df
