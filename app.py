import os
import kagglehub
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Page Configuration
st.set_page_config(page_title="Health Risk Predictor", page_icon="🏥", layout="centered")

# Load external CSS file into Streamlit
with open("styling.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


st.title("🏥 Lifestyle-Based Disease Risk Predictor")
st.write("Enter your daily habits below to calculate your health risk score and get instant preventive advice.")

# 1. Load Data & Train Model (Cached so it only runs once)
@st.cache_resource
def load_and_train_model():
    path = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
    df = pd.read_csv(os.path.join(path, "Sleep_health_and_lifestyle_dataset.csv"))
    
    df['Risk'] = df['Sleep Disorder'].notna().astype(int)
    features = ['Age', 'Sleep Duration', 'Quality of Sleep', 'Physical Activity Level', 'Stress Level', 'Daily Steps']
    
    X = df[features]
    y = df['Risk']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train)
    return model, features

with st.spinner("Loading dataset and training model... Please wait."):
    model, features = load_and_train_model()

st.success("Model trained successfully!")

# 2. Front-End User Input Form
st.header("📋 Patient Habit Input Panel")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    sleep_hrs = st.number_input("Sleep Duration (Hours)", min_value=1.0, max_value=12.0, value=7.0, step=0.5)
    sleep_quality = st.slider("Quality of Sleep (Scale 1-10)", min_value=1, max_value=10, value=7)

with col2:
    exercise_mins = st.number_input("Physical Activity (Minutes/day)", min_value=0, max_value=300, value=30)
    stress = st.slider("Stress Level (Scale 1-10)", min_value=1, max_value=10, value=5)
    steps = st.number_input("Daily Steps", min_value=500, max_value=25000, value=6000, step=500)

# 3. Predict Button & Results Display
if st.button("Calculate Health Risk", type="primary"):
    # Pack data into a DataFrame
    user_data = pd.DataFrame([[age, sleep_hrs, sleep_quality, exercise_mins, stress, steps]], columns=features)
    
    # Predict probability
    prob = model.predict_proba(user_data)[0][1] * 100
    
    st.markdown("---")
    st.subheader("📊 Assessment Report")
    
    # Display Risk Percentage Metric
    st.metric(label="Calculated Disease Risk Probability", value=f"{prob:.1f}%")
    
    # Traffic Light System UI
    if prob >= 70:
        st.error("🔴 **RED ZONE: High Risk** — Immediate lifestyle changes required! Focus on reducing stress, increasing exercise, and improving sleep quality.")
    elif prob >= 35:
        st.warning("🟡 **YELLOW ZONE: Moderate Risk** — Preventive action advised. Consider optimizing your daily routine and physical activity.")
    else:
        st.success("🟢 **GREEN ZONE: Low Risk** — Excellent lifestyle habits! Keep maintaining your current routine.")
        