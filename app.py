import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

st.set_page_config(page_title="Diabetes Risk Predictor", page_icon="🩺", layout="wide")

st.title("🩺 Diabetes Risk Predictor")

# ── Load models from disk ─────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

try:
    with open(os.path.join(BASE_DIR, "scaler.pkl"), "rb") as f:
        sc = pickle.load(f)
    with open(os.path.join(BASE_DIR, "logistic_regression_model.pkl"), "rb") as f:
        model_lr = pickle.load(f)
    with open(os.path.join(BASE_DIR, "tuned_knn_model.pkl"), "rb") as f:
        model_knn = pickle.load(f)
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}. Make sure all .pkl files are in the same folder as app.py.")
    st.stop()

# ── Sidebar: model selector ───────────────────────────────────────────────────

with st.sidebar:
    st.header("🤖 Select Model")
    model_choice = st.radio("Classifier", ["Logistic Regression", "KNN (GridSearchCV Tuned)"])

model = model_lr if model_choice == "Logistic Regression" else model_knn
st.success(f"✅ **{model_choice}** ready")

# ── Prediction form ───────────────────────────────────────────────────────────

st.subheader("Enter Patient Details")
col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies",               min_value=0,   max_value=20,   value=1)
    glucose     = st.number_input("Glucose (mg/dL)",           min_value=0,   max_value=300,  value=110)
    bp          = st.number_input("Blood Pressure (mm Hg)",    min_value=0,   max_value=200,  value=72)
    skin        = st.number_input("Skin Thickness (mm)",       min_value=0,   max_value=100,  value=20)

with col2:
    insulin     = st.number_input("Insulin (IU/mL)",           min_value=0,   max_value=900,  value=80)
    bmi         = st.number_input("BMI",                       min_value=0.0, max_value=70.0, value=25.0, step=0.1)
    dpf         = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0,  value=0.5,  step=0.01)
    age         = st.number_input("Age",                       min_value=1,   max_value=120,  value=30)

if st.button("🔍 Predict Risk", use_container_width=True):
    raw = {
        "Pregnancies": pregnancies, "Glucose": glucose,
        "BloodPressure": bp,        "SkinThickness": skin,
        "Insulin": insulin,         "BMI": bmi,
        "DiabetesPedigreeFunction": dpf, "Age": age
    }
    x_input = sc.transform(pd.DataFrame([raw]))
    pred    = model.predict(x_input)[0]
    proba   = model.predict_proba(x_input)[0]

    st.markdown("---")
    if pred == 1:
        st.error(f"⚠️ **High Risk of Diabetes** — Confidence: {proba[1]:.1%}")
    else:
        st.success(f"✅ **Low Risk of Diabetes** — Confidence: {proba[0]:.1%}")

    st.progress(float(proba[1]), text=f"Diabetic probability: {proba[1]:.1%}")
