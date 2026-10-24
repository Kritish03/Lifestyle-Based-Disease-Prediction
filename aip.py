import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Loading Kaggle Dataset for Sleep Health and Lifestyle

print("Fetching Kaggle Lifestyle Dataset...")
dataset_folder = kagglehub.dataset_download("uom190346a/sleep-health-and-lifestyle-dataset")
csv_path = os.path.join(dataset_folder, "Sleep_health_and_lifestyle_dataset.csv")

df = pd.read_csv(csv_path)
print(f"Dataset Loaded! Total Patients: {len(df)}\n")

# Data Preprocessing

# Convert 'Sleep Disorder' into Binary Risk: 0 = Healthy (Low Risk), 1 = Disorder (High Risk)
df['Disease_Risk'] = df['Sleep Disorder'].notna().astype(int)

# Standardize BMI Category text
df['BMI Category'] = df['BMI Category'].replace({'Normal Weight': 'Normal'})

# Focused Lifestyle Features (Exercise, Diet/BMI, Sleep, Stress)
feature_cols = [
    'Age',
    'Sleep Duration',          
    'Quality of Sleep',        
    'Physical Activity Level', 
    'Stress Level',            # Scale 1-10
    'BMI Category',            # Categorical: Normal, Overweight, Obese
    'Daily Steps'              # count of steps walked daily
]

X = df[feature_cols].copy()
y = df['Disease_Risk']

# Encode 'BMI Category' text into numbers (Normal=0, Obese=1, Overweight=2)
le_bmi = LabelEncoder()
X['BMI Category'] = le_bmi.fit_transform(X['BMI Category'])

# Split into 80% Training and 20% Testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical scales
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

#train model

model = LogisticRegression(C=0.8, max_iter=500, solver='liblinear')
model.fit(X_train_scaled, y_train)

# accuracy 

y_pred = model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("=" * 50)
print(f"MODEL ACCURACY SCORE: {accuracy * 100:.2f}%")
print("=" * 50)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

#ploting bar graph 

plt.figure(figsize=(9, 5))                                                                 
importance = model.coef_[0]
colors = ['#e74c3c' if c > 0 else '#2ecc71' for c in importance]
plt.bar(feature_cols, importance, color=colors)
plt.axhline(0, color='black', linewidth=0.8, linestyle='--')
plt.title("Impact of Lifestyle Habits on Disease Risk (Logistic Regression)")
plt.ylabel("Impact Weight (Red = Increases Risk | Green = Decreases Risk)")
plt.xticks(rotation=30, ha='right')
plt.tight_layout()
print("\n[INFO] Feature Importance chart opened! You can now enter patient data below:")
plt.show(block=False)
plt.pause(1)

def predict_patient_risk(age, sleep_hrs, sleep_quality, exercise_mins, stress, bmi_category, steps):   # defining the predict patient risk function
    patient_df = pd.DataFrame([{
        'Age': age,
        'Sleep Duration': sleep_hrs,
        'Quality of Sleep': sleep_quality,
        'Physical Activity Level': exercise_mins,
        'Stress Level': stress,
        'BMI Category': le_bmi.transform([bmi_category])[0],
        'Daily Steps': steps
    }])
    
    patient_scaled = scaler.transform(patient_df)                       #tells us the risk percentage of the patient based on the model
    prob = model.predict_proba(patient_scaled)[0][1] * 100
    
    #light classification of risk zones 

    if prob < 35:
        zone = "GREEN ZONE (Low Risk - Healthy Lifestyle)"
        print(f"Status Color: #2ecc71")
    elif prob < 70:
        zone = "YELLOW ZONE (Moderate Risk - Preventive Action Advised)"
        print(f"Status Color: #f1c40f")
    else:
        zone = "RED ZONE (High Risk - Immediate Habit Changes Required)"
        print(f"Status Color: #e74c3c")

    print("\n" + "=" * 55)
    print("       PREVENTIVE HEALTHCARE RISK REPORT")
    print("=" * 55)
    print(f"Risk Probability : {prob:.1f}%")
    print(f"Health Status    : {zone}")
    print("-" * 55)
    print("Personalized Preventive Plan:")
    
    tips_given = False
    if exercise_mins < 45 or steps < 6000:
        print(" -> [EXERCISE]: Boost daily workout to 45+ mins and walk at least 7,000 steps/day.")
        tips_given = True
    if bmi_category in ['Overweight', 'Obese']:
        print(" -> [DIET]: Shift to high-protein, high-fiber meals and cut refined sugar to reach Normal BMI.")
        tips_given = True
    if sleep_hrs < 7.0 or sleep_quality < 7:
        print(" -> [SLEEP]: Aim for 7.5 hours of uninterrupted sleep; avoid screens 1 hour before bed.")
        tips_given = True
    if stress > 6:
        print(" -> [STRESS]: High stress spikes cortisol. Add 15 mins of breathing/stretching daily.")
        tips_given = True
    if not tips_given:
        print(" -> [OPTIMAL]: All your lifestyle habits are in the healthy range! Keep it up.")
    print("=" * 55)

#user input

while True:
    print("\n--- LIVE PATIENT RISK CHECKER ---")
    try:
        u_age = int(input("1. Enter Age : "))
        u_sleep = float(input("2. Sleep Hours per night : "))
        u_quality = int(input("3. Sleep Quality [1-10] : "))
        u_exercise = int(input("4. Daily Exercise in minutes : "))
        u_stress = int(input("5. Stress Level [1-10] : "))
        
        # Validate Diet/BMI input
        u_bmi = input("6. Diet/BMI Status (Normal / Overweight / Obese): ").strip().title()
        if u_bmi not in ['Normal', 'Overweight', 'Obese']:
            print("   -> Defaulting to 'Normal' since input wasn't recognized.")
            u_bmi = 'Normal'
            
        u_steps = int(input("7. Daily Steps walked (e.g., 5000): "))

        #prediction of user input 
        predict_patient_risk(u_age, u_sleep, u_quality, u_exercise, u_stress, u_bmi, u_steps)

    except ValueError:
        print("\n[!] Error: Please enter valid numbers for Age, Sleep, Exercise, Stress, and Steps.")

    #asking for the checkup of more patients or not
    again = input("\nDo you want to check another patient? (yes/no): ").strip().lower()
    if again not in ['yes', 'y']:
        print("Exiting Preventive Healthcare System. Stay healthy!")
        break