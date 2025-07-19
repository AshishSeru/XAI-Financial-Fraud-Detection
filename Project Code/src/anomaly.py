from sklearn.ensemble import IsolationForest
import pandas as pd
import joblib
from src.config import ISOLATION_MODEL_PATH, CONTAMINATION, RANDOM_STATE, FLAGGED_TRANSACTIONS_PATH

def run_anomaly_detection(df):
    print("[INFO] Running Isolation Forest anomaly detection...")

    # Drop non-feature columns (IDs or anything irrelevant)
    drop_columns = ['TransactionID', 'AccountID', 'PreviousTransactionDate']
    feature_df = df.drop(columns=drop_columns, errors='ignore')

    # Select only numeric features to avoid errors
    feature_df = feature_df.select_dtypes(include=['float64', 'int64'])

    # Create and fit Isolation Forest
    iso_model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE
    )

    iso_model.fit(feature_df)

    # Add anomaly scores and predictions back to the original df
    df['AnomalyScore'] = iso_model.decision_function(feature_df)
    df['AnomalyFlag'] = iso_model.predict(feature_df)  # -1 = anomaly, 1 = normal
    df['PotentialFraud'] = df['AnomalyFlag'].apply(lambda x: 1 if x == -1 else 0)

    # Save model
    joblib.dump(iso_model, ISOLATION_MODEL_PATH)
    print(f"[INFO] Isolation Forest model saved at {ISOLATION_MODEL_PATH}")

    # Export flagged transactions for review
    flagged = df[df['PotentialFraud'] == 1]
    flagged.to_csv(FLAGGED_TRANSACTIONS_PATH, index=False)
    print(f"[INFO] Flagged transactions exported to {FLAGGED_TRANSACTIONS_PATH}")

    # Remove the AnomalyFlag column from the augmented dataset
    df = df.drop(columns=['AnomalyFlag'], errors='ignore')

    # Export augmented dataset for classifier training
    augmented_dataset_path = 'outputs/augmented_dataset.csv'
    df.to_csv(augmented_dataset_path, index=False)
    print(f"[INFO] Augmented dataset exported to {augmented_dataset_path}")

    return df
