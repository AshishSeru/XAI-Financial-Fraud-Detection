from joblib import load

# Path to the saved model
model_path = "models/fraud_classifier.pkl"

try:
    # Load the model
    model = load(model_path)
    
    # Check if the model supports predict_proba
    if hasattr(model, 'predict_proba'):
        print("[DEBUG] Model supports predict_proba.")
    else:
        print("[DEBUG] Model does NOT support predict_proba.")

except Exception as e:
    print(f"[ERROR] Failed to load or test the model: {e}")