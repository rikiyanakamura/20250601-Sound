
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, train_test_split
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
df = df.dropna()
X = df.drop("metastasis", axis=1)
y = df["metastasis"]

# One-hot encoding for categorical variables
X_encoded = pd.get_dummies(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Display performance
st.subheader("Model Evaluation")
st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
st.write(f"AUC: {roc_auc_score(y_test, y_prob):.2f}")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
fig, ax = plt.subplots()
ax.plot(fpr, tpr, label="ROC curve")
ax.plot([0, 1], [0, 1], "k--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve")
st.pyplot(fig)

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
    prediction = model.predict(input_df_encoded)
    probability = model.predict_proba(input_df_encoded)[0][1]
    st.write(f"Prediction: {'Metastasis' if prediction[0] == 1 else 'No Metastasis'}")
    st.write(f"Probability of Metastasis: {probability:.2f}")
