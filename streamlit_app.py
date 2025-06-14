import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
import matplotlib.pyplot as plt

# Title
st.title("Lymph Node Metastasis Prediction App (Random Forest Model)")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("Train_data.csv")

df = load_data()

# Preprocessing
d = df.dropna()
X = d.drop("metastasis", axis=1)
y = d["metastasis"]

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)

# Display performance
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")

# Use multi-class AUC calculation
auc = roc_auc_score(y_test, y_prob, multi_class="ovr")
st.write(f"AUC: {auc:.2f}")

# ROC Curve (only plot ROC for each class if necessary)
# Optionally add ROC curves for each class later if needed

# Feature importance
st.subheader("Feature Importance")
importances = pd.Series(model.feature_importances_, index=X_encoded.columns)
importances = importances.sort_values(ascending=False)
st.bar_chart(importances)

# Prediction form
st.subheader("Predict from User Input")

def user_input_features():
    input_data = {}
    for column in X.columns:
        val = st.text_input(f"{column}", "")
        input_data[column] = val
    return pd.DataFrame([input_data])

input_df = user_input_features()

if not input_df.isnull().values.any():
    input_df_encoded = pd.get_dummies(input_df)
    input_df_encoded = input_df_encoded.reindex(columns=X_encoded.columns, fill_value=0)
    prediction = model.predict_proba(input_df_encoded)[0]
    for i, prob in enumerate(prediction):
        st.success(f"Predicted probability of class {i}: {prob:.2%}")
