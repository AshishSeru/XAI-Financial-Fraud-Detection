DATA_PATH = 'data/transactions.csv'

# Models
ISOLATION_MODEL_PATH = 'models/isolation_forest.pkl'
CLASSIFIER_MODEL_PATH = 'models/fraud_classifier.pkl'

# Outputs
FLAGGED_TRANSACTIONS_PATH = 'outputs/flagged_transactions.csv'
CLASSIFICATION_RESULTS_PATH = 'outputs/classification_results.csv'
SHAP_OUTPUT_PATH = 'outputs/shap_summary.png'

# Parameters
RANDOM_STATE = 42
CONTAMINATION = 0.05  # Percentage of data flagged as anomalies
TEST_SIZE = 0.2
