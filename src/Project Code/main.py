from src.preprocessing import load_data, preprocess_data
from src.anomaly import run_anomaly_detection
from src.classifier import train_classifier
from src.explainability import shap_explanation_classifier

def main():
    print("========== XAI Powered Fraud Detection Hybrid Pipeline ==========")

    # Step 1: Load and preprocess the data
    df = load_data()
    df_processed = preprocess_data(df)

    # Step 2: Run anomaly detection and add AnomalyScore column
    df_with_anomalies = run_anomaly_detection(df_processed)

    # Step 3: Train classifier (RandomForest/XGBoost) using AnomalyScore as a feature
    classifier_model, feature_df = train_classifier(df_with_anomalies)

    # Step 4: Explain classifier decisions using SHAP
    shap_explanation_classifier(feature_df)

    # Debugging: Verify if the model supports predict_proba
    from joblib import load
    model_path = "models/fraud_classifier.pkl"
    model = load(model_path)
    if hasattr(model, 'predict_proba'):
        print("[DEBUG] Model supports predict_proba.")
    else:
        print("[DEBUG] Model does NOT support predict_proba.")

    print("========== Hybrid Fraud Detection Pipeline Completed ==========")

if __name__ == "__main__":
    main()
