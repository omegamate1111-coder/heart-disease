import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Risk Prediction", layout="wide")
st.title("🫀 Heart Disease Risk Prediction")

# --- Sidebar for inputs ---
st.sidebar.header("Patient Information")

age = st.sidebar.slider("Age", 20, 100, 50)
sex = st.sidebar.selectbox("Sex", ["Female (0)", "Male (1)"])
cp = st.sidebar.selectbox("Chest Pain Type", ["Typical Angina (0)", "Atypical Angina (1)", 
                                               "Non-anginal Pain (2)", "Asymptomatic (3)"])
bp = st.sidebar.slider("Resting Blood Pressure", 80, 200, 120)
chol = st.sidebar.slider("Cholesterol", 100, 600, 200)
fbs = st.sidebar.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No (0)", "Yes (1)"])
ecg = st.sidebar.selectbox("Resting ECG", ["Normal (0)", "ST-T Abnormality (1)", "Left Ventricular Hypertrophy (2)"])
max_hr = st.sidebar.slider("Max Heart Rate", 60, 220, 150)
angina = st.sidebar.selectbox("Exercise Induced Angina", ["No (0)", "Yes (1)"])
oldpeak = st.sidebar.number_input("Oldpeak (ST depression)", 0.0, 10.0, 1.0, 0.1)
slope = st.sidebar.selectbox("ST Slope", ["Upsloping (0)", "Flat (1)", "Downsloping (2)"])

# Map inputs to numeric
sex_val = 0 if sex.startswith("F") else 1
cp_val = ["Typical Angina (0)", "Atypical Angina (1)", "Non-anginal Pain (2)", "Asymptomatic (3)"].index(cp)
fbs_val = 0 if fbs.startswith("N") else 1
ecg_val = ["Normal (0)", "ST-T Abnormality (1)", "Left Ventricular Hypertrophy (2)"].index(ecg)
angina_val = 0 if angina.startswith("N") else 1
slope_val = ["Upsloping (0)", "Flat (1)", "Downsloping (2)"].index(slope)

# --- Prediction ---
if st.button("Predict Risk"):
    # Use exact column names from training
    input_df = pd.DataFrame([[age, sex_val, cp_val, bp, chol, fbs_val, ecg_val, max_hr, angina_val, oldpeak, slope_val]],
                            columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS',
                                     'RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'])
    
    # Scale inputs
    scaled = scaler.transform(input_df)
    prob = model.predict_proba(scaled)[0][1]

    # Determine risk
    if prob < 0.3:
        risk = "Low Risk"
        color = "green"
    elif prob < 0.7:
        risk = "Medium Risk"
        color = "orange"
    else:
        risk = "High Risk"
        color = "red"

    # Display results
    st.subheader(f"Risk Level: {risk}")
    st.markdown(f"<h3 style='color:{color}'>Probability: {prob:.2f}</h3>", unsafe_allow_html=True)

    # Probability Bar Chart
    fig, ax = plt.subplots()
    ax.bar(["No Disease", "Heart Disease"], [1-prob, prob], color=["green","red"])
    ax.set_ylim(0,1)
    ax.set_ylabel("Probability")
    st.pyplot(fig)

    # Optional: Show input data
    with st.expander("View Input Data"):
        st.dataframe(input_df)
