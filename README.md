# Customer Churn Prediction

Predict whether a telecom customer will churn using machine learning.

## Live Demo
(https://churn-prediction-276fanpncsvtjeuopguivf.streamlit.app)

## Dataset
Kaggle Telco Customer Churn — 7043 real customers, 21 features

## Results
| Model | Accuracy | F1 Score | Recall |
|-------|----------|----------|--------|
| Logistic Regression | 0.789 | 0.532 | 0.452 |
| Decision Tree | 0.789 | 0.493 | 0.388 |
| Random Forest | 0.764 | 0.600 | 0.666 |

**Best Model:** Random Forest with class_weight balanced

## What I Built
- Real data cleaning (TotalCharges fix, encoding)
- Feature engineering (avg_monthly_spend)
- Feature selection using RF importance
- 3 model comparison with F1 evaluation
- Threshold tuning for business tradeoff
- Confusion matrix analysis
- Cross-validation (F1: 0.592 ± 0.010)
- Streamlit web app with live predictions

## How to Run

```bash
pip install -r requirements.txt
python churn_model.py
python -m streamlit run app.py
```
## Tech Stack
Python · scikit-learn · Pandas · NumPy · Streamlit
