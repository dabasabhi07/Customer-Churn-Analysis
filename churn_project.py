# ==========================================
# CUSTOMER CHURN PROJECT (SQL + EDA + ML)
# CLEAN & FIXED VERSION
# ==========================================

import sqlite3
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("churn.csv")

conn = sqlite3.connect("churn.db")
df.to_sql("customers", conn, if_exists="replace", index=False)

print("Database created successfully!")

# SQL ANALYSIS
print("\n--- SQL ANALYSIS ---")

print(pd.read_sql("SELECT Churn, COUNT(*) FROM customers GROUP BY Churn", conn))
print(pd.read_sql("SELECT AVG(MonthlyCharges) FROM customers WHERE Churn='Yes'", conn))
print(pd.read_sql("SELECT * FROM customers WHERE tenure < 12 AND MonthlyCharges > 70 LIMIT 5", conn))

# EDA
print("\n--- DATA INFO ---")
print(df.info())

print("\n--- STATISTICS ---")
print(df.describe())

print("\n--- MISSING VALUES ---")
print(df.isnull().sum())

# Target distribution
sns.countplot(x='Churn', data=df)
plt.title("Churn Distribution")
plt.show()

# Numerical distributions
sns.histplot(df['MonthlyCharges'], kde=True)
plt.title("Monthly Charges Distribution")
plt.show()

# Numerical vs churn
sns.boxplot(x='Churn', y='MonthlyCharges', data=df)
plt.title("Monthly Charges vs Churn")
plt.show()

sns.boxplot(x='Churn', y='tenure', data=df)
plt.title("Tenure vs Churn")
plt.show()

# PREPROCESSING
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

df['TotalCharges'] = df['TotalCharges'].fillna(df['TotalCharges'].median())

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

df = df.drop(['customerID'], axis=1)

le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col].astype(str))

# CORRELATION HEATMAP
plt.figure(figsize=(12, 8))

sns.heatmap(df.corr(), 
            cmap='coolwarm', 
            linewidths=0.5)

plt.title("Correlation Heatmap")
plt.show()

# MODELING
X = df.drop('Churn', axis=1)
y = df['Churn']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

lr = LogisticRegression(max_iter=2000)
lr.fit(X_train, y_train)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

lr_pred = lr.predict(X_test)
rf_pred = rf.predict(X_test)

print("\n--- MODEL ACCURACY ---")
print("Logistic Regression:", accuracy_score(y_test, lr_pred))
print("Random Forest:", accuracy_score(y_test, rf_pred))

# CONFUSION MATRIX
cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(cm, annot=True, fmt='d')
plt.title("Confusion Matrix (Random Forest)")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

print("\n--- CLASSIFICATION REPORT ---")
print(classification_report(y_test, rf_pred))

# FEATURE IMPORTANCE
importances = rf.feature_importances_
feature_names = X.columns

feat_imp = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
}).sort_values(by='Importance', ascending=False)

print("\n--- TOP FEATURES ---")
print(feat_imp.head(10))

sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
plt.title("Top 10 Important Features")
plt.show()

conn.close()
