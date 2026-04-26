# Credit Card Fraud Detection

## Overview
End-to-end ML project to detect fraudulent credit card transactions
on a highly imbalanced dataset of 284,807 transactions.

## Key Highlights
- Handled severe class imbalance (0.17% fraud) using SMOTE
- Trained and compared 3 models: Logistic Regression, Random Forest, XGBoost
- Best model: XGBoost with ROC-AUC of 0.9792
- Applied SHAP explainability to interpret model predictions

## Tech Stack
Python, Pandas, Scikit-learn, Imbalanced-learn, XGBoost, SHAP, Matplotlib, Seaborn

## Results
| Model | ROC-AUC |
|-------|---------|
| Logistic Regression | 0.9698 |
| Random Forest | 0.9731 |
| XGBoost | 0.9792 |

## Project Structure
- `fraud_detection.ipynb` — Main notebook
- `confusion_matrices.png` — Model comparison
- `shap_importance.png` — Feature importance
- `shap_dot.png` — SHAP impact plot