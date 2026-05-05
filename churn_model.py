import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, roc_auc_score
)

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

df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# STEP 4: Feature engineering
df["avg_monthly_spend"] = df["TotalCharges"] / (df["tenure"] + 1)

# STEP 5: Encoding
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = LabelEncoder().fit_transform(df[column].astype(str))

# STEP 6
X = df.drop('Churn', axis=1)
y = df['Churn']

# STEP 7
X = X.apply(pd.to_numeric, errors='coerce').fillna(0)

temp_rf = RandomForestClassifier(n_estimators=50, random_state=42)
temp_rf.fit(X, y)

importances = pd.Series(temp_rf.feature_importances_, index=X.columns)
top_features = importances[importances > 0.02].index.tolist()
X = X[top_features]

# STEP 8
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 9
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(max_depth=5),
    "Random Forest": RandomForestClassifier(
        n_estimators=100, max_depth=10, class_weight='balanced'
    )
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Stay','Churn'])
    disp.plot()
    plt.savefig(f"{name}_confusion_matrix.png")
    plt.close()

    f1 = f1_score(y_test, y_pred)
    results[name] = {"model": model, "pred": y_pred, "f1": f1}

# STEP 10
best_name = max(results, key=lambda x: results[x]["f1"])
best_model = results[best_name]["model"]
best_pred = results[best_name]["pred"]

print(f"\nBEST MODEL: {best_name}")
print(classification_report(y_test, best_pred))

# ROC-AUC
y_proba = best_model.predict_proba(X_test_scaled)[:,1]
roc = roc_auc_score(y_test, y_proba)
print(f"ROC-AUC: {roc:.3f}")

# FEATURE IMPORTANCE
if best_name == "Random Forest":
    feat_imp = pd.Series(best_model.feature_importances_, index=X.columns)
    feat_imp = feat_imp.sort_values(ascending=False).head(10)

    print("\nTop Features:")
    print(feat_imp)

    plt.figure()
    feat_imp.plot(kind='barh')
    plt.gca().invert_yaxis()
    plt.savefig("feature_importance.png")
    plt.close()

# STEP 11
proba = best_model.predict_proba(X_test_scaled)[:,1]

# STEP 12
cv = cross_val_score(best_model, X, y, cv=5, scoring='f1')
print(f"CV F1: {cv.mean():.3f}")

# SAVE
import pickle
pickle.dump(best_model, open("model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
pickle.dump(top_features, open("features.pkl","wb"))

print("PROJECT COMPLETE")
