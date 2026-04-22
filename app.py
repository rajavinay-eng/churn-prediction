import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ─────────────────────────────────────────────────────────────
# Load saved model, scaler, and feature list
# ─────────────────────────────────────────────────────────────
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("features.pkl", "rb") as f:
    top_features = pickle.load(f)

# ─────────────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Predictor",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.markdown("Predict whether a telecom customer will leave — "
            "built with real Kaggle data and Random Forest")

# ─────────────────────────────────────────────────────────────
# Sidebar inputs — customer details
# ─────────────────────────────────────────────────────────────
st.sidebar.header("Enter Customer Details")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges (CHF)", 0, 150, 70)
total_charges = monthly_charges * tenure
avg_monthly_spend = total_charges / (tenure + 1)

senior_citizen = st.sidebar.selectbox("Senior Citizen", [0, 1])
partner = st.sidebar.selectbox("Has Partner", ["No", "Yes"])
dependents = st.sidebar.selectbox("Has Dependents", ["No", "Yes"])
contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)
paperless = st.sidebar.selectbox("Paperless Billing", ["No", "Yes"])

# ─────────────────────────────────────────────────────────────
# Convert inputs to numbers
# ─────────────────────────────────────────────────────────────
partner_num = 1 if partner == "Yes" else 0
dependents_num = 1 if dependents == "Yes" else 0
contract_num = {"Month-to-month": 0, "One year": 1, "Two year": 2}[contract]
paperless_num = 1 if paperless == "Yes" else 0

# ─────────────────────────────────────────────────────────────
# Build input dataframe matching model features
# ─────────────────────────────────────────────────────────────
input_data = {
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "avg_monthly_spend": avg_monthly_spend,
    "SeniorCitizen": senior_citizen,
    "Partner": partner_num,
    "Dependents": dependents_num,
    "Contract": contract_num,
    "PaperlessBilling": paperless_num,
}

input_df = pd.DataFrame([input_data])

# Keep only features the model was trained on
for feature in top_features:
    if feature not in input_df.columns:
        input_df[feature] = 0

input_df = input_df[top_features]
input_scaled = scaler.transform(input_df)

# ─────────────────────────────────────────────────────────────
# Prediction
# ─────────────────────────────────────────────────────────────
proba = model.predict_proba(input_scaled)[0][1]
prediction = "WILL CHURN" if proba > 0.4 else "WILL STAY"

# ─────────────────────────────────────────────────────────────
# Display results
# ─────────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Churn Probability", f"{proba*100:.1f}%")

with col2:
    st.metric("Tenure", f"{tenure} months")

with col3:
    st.metric("Monthly Charges", f"CHF {monthly_charges}")

st.markdown("---")

if proba > 0.6:
    st.error(f"🚨 HIGH RISK — {prediction} (Probability: {proba*100:.1f}%)")
    st.write("**Action:** Call this customer immediately. Offer 20% discount or contract upgrade.")
elif proba > 0.4:
    st.warning(f"⚠️ MEDIUM RISK — {prediction} (Probability: {proba*100:.1f}%)")
    st.write("**Action:** Send personalised retention email with loyalty offer.")
else:
    st.success(f"✅ LOW RISK — {prediction} (Probability: {proba*100:.1f}%)")
    st.write("**Action:** No immediate action needed. Monitor quarterly.")

st.markdown("---")
st.subheader("Customer Summary")
summary = {
    "Tenure": f"{tenure} months",
    "Monthly Charges": f"CHF {monthly_charges}",
    "Total Charges": f"CHF {total_charges}",
    "Contract": contract,
    "Senior Citizen": "Yes" if senior_citizen == 1 else "No"
}
st.table(pd.DataFrame(summary.items(), columns=["Feature", "Value"]))

st.markdown("---")
st.caption("Model: Random Forest | Dataset: Kaggle Telco Customer Churn | "
           "7043 customers | F1: 0.60 | Recall: 0.67")