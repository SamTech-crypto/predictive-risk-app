import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
import shap

# App Title
st.title("🔍 Predictive Risk Modeling for Loan Default")

# Simulate Data
np.random.seed(42)
n_samples = 1000
data = pd.DataFrame({
    'credit_score': np.random.randint(300, 850, n_samples),
    'income': np.random.randint(30000, 100000, n_samples),
    'debt_to_income': np.random.uniform(0, 1, n_samples),
    'loan_amount': np.random.randint(5000, 50000, n_samples),
    'default_status': np.random.choice([0, 1], size=n_samples, p=[0.7, 0.3])
})
data['income_to_loan_ratio'] = data['income'] / data['loan_amount']

# Data Preview
st.subheader("📊 Data Preview")
st.dataframe(data.head())

# Feature Distribution
st.subheader("📈 Credit Score Distribution")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data['credit_score'], kde=True, bins=30, ax=ax)
st.pyplot(fig)

# Prepare Data for Modeling
X = data[['credit_score', 'income', 'debt_to_income', 'loan_amount', 'income_to_loan_ratio']]
y = data['default_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame for SHAP compatibility
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)

# Show Metrics
st.subheader("📉 Model Performance")
st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
st.write(f"**ROC-AUC:** {roc_auc_score(y_test, y_pred):.2f}")

# SHAP Explainability
st.subheader("🔎 SHAP Explainability")

# Use TreeExplainer for RandomForestClassifier
explainer = shap.TreeExplainer(model)

# Compute SHAP values with feature names preserved
shap_values = explainer.shap_values(X_test_scaled_df)

# SHAP Summary Plot for the positive class (1)
shap.summary_plot(shap_values[1], X_test_scaled_df, show=False)
st.pyplot(plt.gcf())
