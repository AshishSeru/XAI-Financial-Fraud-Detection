from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, roc_auc_score, precision_score, recall_score, f1_score
import joblib
from src.config import CLASSIFIER_MODEL_PATH, TEST_SIZE, RANDOM_STATE, CLASSIFICATION_RESULTS_PATH
from tabulate import tabulate

def train_classifier(df):
    print("[INFO] Training classifier (RandomForest)...")

    # Drop ID and date columns, as they are not useful for classification
    drop_cols = ['TransactionID', 'AccountID', 'PreviousTransactionDate', 'PotentialFraud', 'AnomalyFlag']
    feature_df = df.drop(columns=drop_cols, errors='ignore')

    # Ensure only numeric data is passed to the model
    feature_df = feature_df.select_dtypes(include=['float64', 'int64'])

    X = feature_df
    y = df['PotentialFraud']

    # Perform k-fold cross-validation
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []

    for fold, (train_idx, test_idx) in enumerate(kfold.split(X, y)):
        print(f"[INFO] Processing fold {fold + 1}...")

        # Split data into training and testing sets for this fold
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train the classifier
        clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
        clf.fit(X_train, y_train)

        # Evaluate the model
        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]

        # Save actual vs predicted output to a CSV file
        fold_output = X_test.copy()
        fold_output['Actual'] = y_test.values
        fold_output['Predicted'] = y_pred
        fold_output['Probability'] = y_proba
        output_path = f"outputs/kfold_result_fold_{fold + 1}.csv"
        fold_output.to_csv(output_path, index=False)
        print(f"[INFO] Kfold result output saved for fold {fold + 1} at {output_path}")

        metrics = {
            "fold": fold + 1,
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba)
        }
        fold_metrics.append(metrics)

        print(f"[INFO] Fold {fold + 1} Metrics: {metrics}")

    # Calculate average metrics across all folds
    avg_metrics = {
        "precision": sum(m["precision"] for m in fold_metrics) / len(fold_metrics),
        "recall": sum(m["recall"] for m in fold_metrics) / len(fold_metrics),
        "f1": sum(m["f1"] for m in fold_metrics) / len(fold_metrics),
        "roc_auc": sum(m["roc_auc"] for m in fold_metrics) / len(fold_metrics)
    }

    # Print metrics for each fold in tabular format
    print("\n[INFO] Metrics for Each Fold:")
    print(tabulate(fold_metrics, headers="keys", tablefmt="grid"))

    # Print average metrics in tabular format
    print("\n[INFO] Average Metrics Across Folds:")
    print(tabulate([avg_metrics], headers="keys", tablefmt="grid"))

    # Save the final model trained on the entire dataset
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE)
    clf.fit(X, y)
    joblib.dump(clf, CLASSIFIER_MODEL_PATH)
    print(f"[INFO] Final classifier model saved at {CLASSIFIER_MODEL_PATH}")

    return clf, X

def predict_fraud(model, input_data):
    """
    Predict fraud probability and classification for input data.

    Args:
        model: Trained classifier model.
        input_data: Preprocessed input data (Pandas DataFrame).

    Returns:
        prob: Fraud probability (float).
        pred: Fraud prediction (int, 1 for fraud, 0 for non-fraud).
    """
    prob = model.predict_proba(input_data)[:, 1][0]  # Probability of fraud
    pred = model.predict(input_data)[0]  # Binary prediction (1 for fraud, 0 for non-fraud)
    return prob, pred
