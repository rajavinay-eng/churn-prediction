import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, classification_report

print("=" * 60)
print("CUSTOMER CHURN PREDICTION — TELCO DATASET")
print("=" * 60)

# STEP 1: Load data
df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
print(f"\nDataset loaded: {df.shape[0]} customers, {df.shape[1]} features")

# STEP 2: Explore
print("\nChurn distribution:")
print(df['Churn'].value_counts())
print("\nChurn percentage:")
print(df['Churn'].value_counts(normalize=True).round(3))

# STEP 3: Clean
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())
df = df.drop('customerID', axis=1)

# 🔥 FIX TARGET
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# STEP 4: Feature engineering
df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

# STEP 5: Encoding
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = df[column].astype(str)
        df[column] = LabelEncoder().fit_transform(df[column])

# STEP 6: Features/target
X = df.drop('Churn', axis=1)
y = df['Churn']

print(f"\nTarget: Stay={sum(y==0)}, Churn={sum(y==1)}")

# STEP 7: Feature selection
X = X.apply(pd.to_numeric, errors='coerce')
X = X.fillna(0)

temp_rf = RandomForestClassifier(n_estimators=50, random_state=42)
temp_rf.fit(X, y)

importances = pd.Series(temp_rf.feature_importances_, index=X.columns)
importances_sorted = importances.sort_values(ascending=False)

print("\nTop important features:")
print(importances_sorted.head(10).round(3))

top_features = importances_sorted[importances_sorted > 0.02].index.tolist()
X = X[top_features]

# STEP 8: Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\nTraining: {len(X_train)} | Testing: {len(X_test)}")

# STEP 9: Models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=42),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10,
        class_weight='balanced', random_state=42
    )
}

results = {}
print("\n" + "=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)

    results[name] = {"model": model, "pred": y_pred, "f1": f1}

    print(f"\n{name}")
    print(f"Accuracy: {acc:.3f} | F1: {f1:.3f}")
    print(f"Precision: {prec:.3f} | Recall: {rec:.3f}")

# STEP 10: Best model
best_name = max(results, key=lambda x: results[x]["f1"])
best_model = results[best_name]["model"]
best_pred = results[best_name]["pred"]

print("\n" + "=" * 60)
print(f"BEST MODEL: {best_name}")
print("=" * 60)

print(classification_report(y_test, best_pred))

cm = confusion_matrix(y_test, best_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nTN:{tn} FP:{fp} FN:{fn} TP:{tp}")
print(f"Recall: {tp/(tp+fn):.2f}")
print(f"Precision: {tp/(tp+fp+1e-6):.2f}")

# STEP 11: Threshold tuning
print("\nThreshold tuning:")
proba = best_model.predict_proba(X_test_scaled)[:, 1]

for t in [0.3, 0.4, 0.5, 0.6]:
    pred = (proba >= t).astype(int)
    print(f"{t} → F1: {f1_score(y_test, pred):.3f}")

# STEP 12: Business output
X_test_df = X_test.copy()
X_test_df['prob'] = proba

high_risk = X_test_df[X_test_df['prob'] > 0.5]
print(f"\nHigh risk customers: {len(high_risk)}")

# STEP 13: Cross-validation
rf_cv = RandomForestClassifier(
    n_estimators=100, max_depth=10,
    class_weight='balanced', random_state=42
)

cv_scores = cross_val_score(rf_cv, X, y, cv=5, scoring='f1')

print(f"\nCV F1: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

print("\nPROJECT COMPLETE")


# SAVE MODEL FILES
import pickle

with open("model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("features.pkl", "wb") as f:
    pickle.dump(top_features, f)

print("Files saved")