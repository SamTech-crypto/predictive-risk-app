import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, roc_curve, confusion_matrix
import shap
import io
from contextlib import redirect_stdout

st.title("🔍 Predictive Risk Modeling for Loan Default")

# Simulate data
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

# Data preview
st.subheader("📊 Data Preview")
st.dataframe(data.head())

# Correlation heatmap
st.subheader("🔗 Feature Correlations")
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(data.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# Credit Score Distribution
st.subheader("📈 Credit Score Distribution by Default Status")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(data=data, x='credit_score', hue='default_status', kde=True, bins=30, ax=ax)
ax.legend(['Non-Default', 'Default'])
st.pyplot(fig)

# Model training
features = ['credit_score', 'income', 'debt_to_income', 'loan_amount', 'income_to_loan_ratio']
X = data[features]
y = data['default_status']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

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
try:
    explainer = shap.Explainer(model, X_train_scaled)
    shap_values = explainer(X_test_scaled)

    fig, ax = plt.subplots(figsize=(10, 6))
    shap.summary_plot(shap_values, X_test, show=False)
    st.pyplot(fig)
except Exception as e:
    st.error(f"Error in SHAP: {str(e)}")

# Interactive Prediction
st.subheader("🛠️ Predict Loan Default")
with st.form("prediction_form"):
    credit_score = st.slider("Credit Score", 300, 850, 600)
    income = st.number_input("Annual Income ($)", 30000, 100000, 50000)
    debt_to_income = st.slider("Debt-to-Income Ratio", 0.0, 1.0, 0.4)
    loan_amount = st.number_input("Loan Amount ($)", 5000, 50000, 20000)
    submitted = st.form_submit_button("Predict")

    if submitted:
        input_df = pd.DataFrame({
            'credit_score': [credit_score],
            'income': [income],
            'debt_to_income': [debt_to_income],
            'loan_amount': [loan_amount],
            'income_to_loan_ratio': [income / loan_amount]
        }, columns=features)

        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        probability = model.predict_proba(input_scaled)[0][1]

        st.write(f"**Prediction:** {'Default' if prediction == 1 else 'Non-Default'}")
        st.write(f"**Probability of Default:** {probability:.2%}")

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
