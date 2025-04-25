import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import shap
import io
from contextlib import redirect_stdout

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

# Correlation Heatmap
st.subheader("🔗 Feature Correlations")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Feature Distribution
st.subheader("📈 Credit Score Distribution by Default Status")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=data, x='credit_score', hue='default_status', kde=True, bins=30, ax=ax)
ax.legend(['Non-Default', 'Default'])
st.pyplot(fig)

# Prepare Data for Modeling
X = data[['credit_score', 'income', 'debt_to_income', 'loan_amount', 'income_to_loan_ratio']]
y = data['default_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]

# Model Performance
st.subheader("📉 Model Performance")
col1, col2 = st.columns(2)
with col1:
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"**Precision:** {precision_score(y_test, y_pred):.2f}")
with col2:
    st.write(f"**Recall:** {recall_score(y_test, y_pred):.2f}")
    st.write(f"**ROC-AUC:** {roc_auc_score(y_test, y_pred_proba):.2f}")

# Confusion Matrix
st.subheader("🔢 Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
st.pyplot(fig)

# ROC Curve
st.subheader("📈 ROC Curve")
fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
fig, ax = plt.subplots(figsize=(8, 6))
ax.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc_score(y_test, y_pred_proba):.2f})')
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend()
st.pyplot(fig)

# SHAP Explainability
st.subheader("🔎 SHAP Explainability")
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# SHAP Summary Plot
st.write("**Feature Importance for Default (Class 1)**")
fig, ax = plt.subplots(figsize=(10, 6))
shap.summary_plot(shap_values[1], X_test, show=False)
st.pyplot(fig)

# SHAP Dependence Plot (for credit_score)
st.write("**Credit Score vs. Debt-to-Income Interaction**")
fig, ax = plt.subplots(figsize=(10, 6))
shap.dependence_plot('credit_score', shap_values[1], X_test, interaction_index='debt_to_income', show=False)
st.pyplot(fig)

# Interactive Prediction
st.subheader("🛠️ Predict Loan Default")
with st.form("prediction_form"):
    credit_score = st.slider("Credit Score", 300, 850, 600)
    income = st.number_input("Annual Income ($)", 30000, 100000, 50000)
    debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)
    loan_amount = st.number_input("Loan Amount ($)", 5000, 50000, 20000)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = pd.DataFrame({
            'credit_score': [credit_score],
            'income': [income],
            'debt_to_income': [debt_to_income],
            'loan_amount': [loan_amount],
            'income_to_loan_ratio': [income / loan_amount]
        })
        pred = model.predict(input_data)[0]
        pred_proba = model.predict_proba(input_data)[0][1]
        st.write(f"**Prediction:** {'Default' if pred == 1 else 'Non-Default'}")
        st.write(f"**Default Probability:** {pred_proba:.2%}")

        # SHAP for Individual Prediction
        shap_vals = explainer.shap_values(input_data)
        st.write("**Why this prediction?**")
        fig, ax = plt.subplots(figsize=(10, 4))
        shap.decision_plot(explainer.expected_value[1], shap_vals[1], input_data, show=False)
        st.pyplot(fig)

# Downloadable Report
st.subheader("📥 Download Report")
report = io.StringIO()
with redirect_stdout(report):
    print("Loan Default Prediction Report")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    print(f"Precision: {precision_score(y_test, y_pred):.2f}")
    print(f"Recall: {recall_score(y_test, y_pred):.2f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, y_pred_proba):.2f}")
st.download_button("Download Report", report.getvalue(), "report.txt")
