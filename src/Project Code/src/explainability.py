import shap
import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.config import CLASSIFIER_MODEL_PATH, SHAP_OUTPUT_PATH

def shap_explanation_classifier(df, model=None):
    print("[INFO] Generating SHAP explanations for classifier...")

    # Use the provided model or load it from the file
    clf = model if model is not None else joblib.load(CLASSIFIER_MODEL_PATH)

    # Ensure the model is a valid scikit-learn or XGBoost model
    if not hasattr(clf, 'predict') or not hasattr(clf, 'predict_proba'):
        raise ValueError(f"Unsupported model type for SHAP TreeExplainer: {type(clf)}")

    # Debugging: Log the type of the model
    print(f"[DEBUG] Model type passed to SHAP TreeExplainer: {type(clf)}")

    # Tree Explainer for RandomForest/XGBoost
    explainer = shap.TreeExplainer(clf)

    # Get shap values (for binary classification you can use shap_values[1])
    shap_values = explainer.shap_values(df)

    # Check if shap_values is a list or array
    if isinstance(shap_values, list):
        shap_to_plot = shap_values[1]  # For class 1 (fraud class)
    else:
        shap_to_plot = shap_values

    # Use SHAP values for the fraud class (class 1)
    if len(shap_to_plot.shape) == 3:
        shap_to_plot = shap_to_plot[:, :, 1]  # Select fraud class

    # Enhanced SHAP summary plot with all features
    plt.title("SHAP Summary Plot - Fraud Classifier (All Features)")
    shap.summary_plot(shap_to_plot, df, max_display=df.shape[1], show=False)
    plt.savefig(SHAP_OUTPUT_PATH)
    print(f"[INFO] SHAP summary plot saved at {SHAP_OUTPUT_PATH}")

    # bar plot for top contributing features
    plt.title("Top 15 Features by SHAP Importance")
    shap.summary_plot(shap_to_plot, df, plot_type="bar", max_display=15, show=False)
    plt.title("Top 15 Features by SHAP Importance")
    plt.savefig("outputs/shap_bar_plot.png")
    print("[INFO] SHAP bar plot saved as outputs/shap_bar_plot.png")

    print("[INFO] Feature importance (mean SHAP value):")
    import numpy as np

    # Debugging: Print shapes of SHAP values and feature names
    print(f"[DEBUG] SHAP values shape: {shap_to_plot.shape}")
    print(f"[DEBUG] Feature names count: {len(df.columns)}")

    # Calculate mean SHAP values for each feature
    mean_shap_values = np.abs(shap_to_plot).mean(axis=0)

    # Print feature importance (mean SHAP value) as a table
    from tabulate import tabulate
    feature_importance_table = [
        {"Feature": feature, "Mean SHAP Value": value}
        for feature, value in zip(df.columns, mean_shap_values)
    ]
    print("\n[INFO] Feature Importance Table:")
    print(tabulate(feature_importance_table, headers="keys", tablefmt="grid"))

    # Save SHAP values per transaction for class 1 (fraud)
    shap_df = pd.DataFrame(shap_to_plot, columns=df.columns)
    shap_df.to_csv("outputs/shap_values_per_transaction.csv", index=False)
    print("[INFO] SHAP values exported to outputs/shap_values_per_transaction.csv")

    # Generate human-readable explanations
    explanations = []
    for feature, shap_value in zip(df.columns, shap_to_plot[0]):
        if shap_value > 0:
            explanations.append(f"{feature} increased the likelihood of fraud.")
        else:
            explanations.append(f"{feature} decreased the likelihood of fraud.")

    # Return the explanations as a DataFrame
    explanation_df = pd.DataFrame({
        "Feature": df.columns,
        "Explanation": explanations
    })
    return explanation_df

