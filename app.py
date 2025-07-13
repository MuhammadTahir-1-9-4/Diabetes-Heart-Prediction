
import streamlit as st
import joblib
import numpy as np
import pandas as pd
from io import StringIO
from datetime import datetime
import matplotlib.pyplot as plt
import pickle

# Load models
heart_model = joblib.load("heart_prediction_model.pkl")

with open('diabetes_prediction_model.pkl', 'rb') as f:
    diabetes_model = pickle.load(f)

# Diabetes model expected columns
diabetes_columns = [
    'age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
    'blood_glucose_level', 'gender_Male', 'smoking_history_ever',
    'smoking_history_former', 'smoking_history_never', 'smoking_history_not current'
]

st.set_page_config(page_title="Health Risk Predictor", layout="wide")
st.markdown("""
    <style>
    .main, .stApp {
        background: linear-gradient(135deg, #f6d365 0%, #fda085 25%, #fbc2eb 50%, #a6c1ee 75%, #f6d365 100%);
        background-attachment: fixed;
    }
    .stButton>button {
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-size: 16px;
        background-color: #4CAF50;
        color: white;
        border: none;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    </style>
""", unsafe_allow_html=True)

st.title("🩺 Health Risk Prediction Dashboard")

# Memory log of predictions
if 'prediction_log' not in st.session_state:
    st.session_state.prediction_log = []

# Real-time KPIs
with st.container():
    total = len(st.session_state.prediction_log)
    heart_count = sum(1 for x in st.session_state.prediction_log if x['type'] == 'Heart')
    diabetes_count = sum(1 for x in st.session_state.prediction_log if x['type'] == 'Diabetes')

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Predictions", total)
    col2.metric("Heart Risk Entries", heart_count)
    col3.metric("Diabetes Entries", diabetes_count)

# Sidebar filter
st.sidebar.header("🔎 Filter Predictions")
selected_type = st.sidebar.selectbox("Select Prediction Type", ["All", "Heart", "Diabetes"])

# Tabs for heart and diabetes
tabs = st.tabs(["❤️ Heart Disease Prediction", "🩸 Diabetes Prediction"])

# --------------------
# HEART DISEASE TAB
# --------------------
with tabs[0]:
    st.header("Heart Disease Risk Assessment")

    age = st.slider("Age", 18, 100, 45, help="Patient's age in years")
    sex = st.selectbox("Sex", ["Male", "Female"], help="Biological sex")

    cp_map = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-anginal Pain": 2,
        "Asymptomatic": 3
    }
    cp_choice = st.selectbox("Chest Pain Type", list(cp_map.keys()), help="Type of chest pain experienced")
    cp = cp_map[cp_choice]

    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 80, 200, 120, help="Resting blood pressure")
    chol = st.number_input("Cholesterol (mg/dl)", 100, 600, 200, help="Serum cholesterol level")
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"], help="Indicates whether fasting blood sugar is high")
    fbs = 1 if fbs == "Yes" else 0

    restecg_map = {
        "Normal": 0,
        "ST-T Wave Abnormality": 1,
        "Left Ventricular Hypertrophy": 2
    }
    restecg_choice = st.selectbox("Resting ECG Result", list(restecg_map.keys()), help="Electrocardiographic results")
    restecg = restecg_map[restecg_choice]

    thalach = st.number_input("Max Heart Rate Achieved", 60, 220, 150, help="Maximum heart rate achieved during test")
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"], help="Presence of exercise-induced chest pain")
    exang = 1 if exang == "Yes" else 0

    oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0, help="ST depression induced by exercise relative to rest")

    slope_map = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    slope_choice = st.selectbox("Slope of ST Segment", list(slope_map.keys()), help="Slope of peak exercise ST segment")
    slope = slope_map[slope_choice]

    ca = st.selectbox("Major Vessels Colored", [0, 1, 2, 3, 4], help="Number of major vessels colored by fluoroscopy")

    thal_map = {
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }
    thal_choice = st.selectbox("Thalassemia", list(thal_map.keys()), help="Type of blood disorder")
    thal = thal_map[thal_choice]

    heart_features = np.array([
        age, 1 if sex == "Male" else 0, cp, trestbps, chol,
        fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ]).reshape(1, -1)

    if st.button("🔍 Predict Heart Disease"):
        pred = heart_model.predict(heart_features)[0]
        risk = "High Risk 💔" if pred == 1 else "Low Risk ❤️"
        st.success(f"Prediction: {risk}")

        st.session_state.prediction_log.append({
            "type": "Heart", "result": risk, "time": datetime.now().isoformat()
        })

# --------------------
# DIABETES TAB
# --------------------
with tabs[1]:
    st.header("Diabetes Risk Assessment")

    age = st.slider("Age", 1, 120, 40, help="Patient's age in years")
    hypertension = st.selectbox("Hypertension", [0, 1], help="0 = No, 1 = Yes")
    heart_disease = st.selectbox("Heart Disease", [0, 1], help="0 = No, 1 = Yes")
    bmi = st.number_input("BMI", 10.0, 60.0, 25.0, help="Body Mass Index")
    hba1c = st.number_input("HbA1c Level", 3.0, 15.0, 5.5, help="Glycated hemoglobin level")
    glucose = st.number_input("Blood Glucose Level", 50, 500, 100, help="Blood glucose level in mg/dl")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Biological sex")
    smoking = st.selectbox("Smoking History", ["never", "former", "ever", "not current"], help="Patient's smoking history")

    input_dict = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c,
        'blood_glucose_level': glucose,
        'gender_Male': 1 if gender == 'Male' else 0,
        'smoking_history_ever': 1 if smoking == 'ever' else 0,
        'smoking_history_former': 1 if smoking == 'former' else 0,
        'smoking_history_never': 1 if smoking == 'never' else 0,
        'smoking_history_not current': 1 if smoking == 'not current' else 0
    }

    input_df = pd.DataFrame([input_dict])[diabetes_columns]

    if st.button("🔍 Predict Diabetes"):
        pred = diabetes_model.predict(input_df)[0]
        risk = "Likely Diabetic ⚠️" if pred == 1 else "Unlikely Diabetic ✅"
        st.success(f"Prediction: {risk}")

        st.session_state.prediction_log.append({
            "type": "Diabetes", "result": risk, "time": datetime.now().isoformat()
        })

# --------------------
# DOWNLOAD SECTION + CHARTS
# --------------------

st.markdown("---")
st.subheader("📄 Prediction Log")

log_data = st.session_state.prediction_log
if selected_type != "All":
    log_data = [log for log in log_data if log['type'] == selected_type]

if log_data:
    df_log = pd.DataFrame(log_data)
    st.dataframe(df_log, use_container_width=True)

    csv = df_log.to_csv(index=False)
    st.download_button("📥 Download Log as CSV", csv, "predictions.csv", "text/csv")

    with st.expander("📊 Show Charts"):
        st.subheader("📊 Prediction Trends")
        df_log['time'] = pd.to_datetime(df_log['time'])
        count_data = df_log.groupby(['type', 'result']).size().unstack().fillna(0)

        if st.checkbox("📈 Show Bar Chart of Predictions"):
            st.bar_chart(count_data)

        if st.checkbox("🥧 Show Pie Chart of Risk Distribution"):
            pie_data = df_log['result'].value_counts()
            fig, ax = plt.subplots()
            ax.pie(pie_data, labels=pie_data.index, autopct='%1.1f%%', startangle=90)
            ax.axis('equal')
            st.pyplot(fig)
else:
    st.info("No predictions made yet. Please use the forms above.")