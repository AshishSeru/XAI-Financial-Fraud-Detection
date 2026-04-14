# XAI-Driven Financial Fraud Detection System

An end-to-end **Machine Learning framework for financial fraud detection** that integrates  
**anomaly detection**, **ensemble learning**, and **Explainable AI (XAI)** to achieve high
accuracy while maintaining **model transparency and auditability**.

This project is based on a **peer-reviewed research publication** and focuses on building
trustworthy, interpretable ML systems suitable for real-world financial environments.

---

## Problem Statement

Financial fraud detection systems face three major challenges:

- Severe **class imbalance** (fraud cases are rare)
- Constantly **evolving fraud patterns**
- Lack of **explainability**, making ML models difficult to trust and regulate

This project addresses these challenges by combining:
- SMOTE-based data balancing
- Anomaly detection models
- Ensemble classifiers
- Explainable AI techniques (SHAP, LIME)

---

## System Architecture

### Pipeline Overview

1. **Data Preprocessing**
   - Data cleaning and feature engineering
   - Class imbalance handling using **SMOTE**

2. **Anomaly Detection Layer**
   - Autoencoder-based reconstruction error
   - Isolation Forest for outlier detection

3. **Classification Layer**
   - Random Forest
   - XGBoost (primary high-performing model)

4. **Explainability Layer**
   - SHAP for global and local feature importance
   - LIME for instance-level explanations

This layered architecture ensures **accuracy, robustness, and interpretability**.

---

## Key Results (Research-Backed)

| Model          | Accuracy | Precision | Recall | F1-Score |
|----------------|----------|-----------|--------|----------|
| Random Forest  | 0.93     | 0.75      | 0.78   | 0.76     |
| XGBoost        | **0.95** | **0.80**  | **0.81** | **0.80** |

- SMOTE improved fraud recall from **0.44 → 0.76**
- XGBoost delivered the best balance between precision and recall
- SHAP enabled transparent feature-level decision tracing

---

## Repository Structure
XAI-Financial-Fraud-Detection/
│
├── src/ # Core ML pipeline
│ ├── preprocessing.py
│ ├── anomaly.py
│ ├── classifier.py
│ ├── explainability.py
│ └── config.py
│
├── data/ # Transaction datasets
├── models/ # Trained models & encoders
├── outputs/ # Evaluation results & plots
│
├── publications/ # Research paper & poster
├── requirements.txt
└── README.md


---

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline:
python src/main.py

Launch the dashboard (if applicable):
python fraud_dashboard.py

📄Research Publication

This implementation is based on the peer-reviewed paper:

“A Machine Learning Framework for Financial Fraud Detection Using Explainable Artificial Intelligence Techniques”
Published in International Journal of Computer Sciences and Engineering (IJCSE), 2025

🔗 DOI: https://doi.org/10.26438/ijcse/v13i5.1725

👥 Authors & Contributions

Ashish Seru – ML pipeline design, model development, evaluation

Archit Mehrotra – Co-developer, experimentation & validation

Tanisha Gotadke – Documentation & research synthesiss

Faculty mentors – Research supervision

This repository represents the engineering implementation of the published research.

🛠 Tech Stack

Python

Scikit-learn

XGBoost, Random Forest

Isolation Forest

SHAP, LIME

Pandas, NumPy

Streamlit

## Why This Project Matters

This project goes beyond a toy ML implementation and reflects real-world financial system requirements:

- **Production-oriented ML design** with modular, maintainable pipeline components  
- **Explainable AI (XAI)** to support transparency and regulatory compliance  
- **Research-backed methodology**, validated through a peer-reviewed publication  
- **Clear separation of ML stages** (preprocessing, modeling, evaluation, explainability)  
- **Industry relevance**, aligned with fraud detection use cases in banking and fintech systems

📬 Contact

Ashish Seru
GitHub: https://github.com/AshishSeru

LinkedIn: https://www.linkedin.com/in/ashishseru/
