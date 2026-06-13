import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Page configuration and layout styling
st.set_page_config(page_title="Obesity Risk Analytics", page_icon="⚖️", layout="centered")

st.title("⚖️ Group 14: Advanced Obesity Risk & Lifestyle Screener")
st.markdown("""
This production-grade screening system pairs machine learning classification with a **Behavioral Risk Engine** to detect latent lifestyle habits before weight gain manifests clinically.
""")
st.write("---")

# 1. Load the exported pipeline artifacts safely
@st.cache_resource
def load_pipeline():
    return joblib.load('obesity_model_pipeline.joblib')

try:
    artifacts = load_pipeline()
    model = artifacts['model']
    scaler = artifacts['scaler']
    trained_features = artifacts['features']
except Exception as e:
    st.error(f"⚠️ Could not load model file. Make sure 'obesity_model_pipeline.joblib' is in this same folder! Error: {e}")
    st.stop()

# 2. User Interface Inputs
st.header("👤 Section 1: Demographics & Physical Attributes")
col1, col2, col3 = st.columns(3)
with col1:
    gender = st.selectbox("Gender", options=["Female", "Male"])
with col2:
    age = st.slider("Age (Years)", min_value=14, max_value=65, value=25, step=1)
with col3:
    family_history = st.selectbox("Family History of Overweight?", options=["yes", "no"])

col4, col5 = st.columns(2)
with col4:
    height = st.number_input("Height (Meters)", min_value=1.40, max_value=2.20, value=1.70, step=0.01)
with col5:
    weight = st.number_input("Weight (Kilograms)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)

st.write("---")
st.header("🥗 Section 2: Dietary & Eating Habits")
col6, col7, col8 = st.columns(3)
with col6:
    favc = st.selectbox("Frequent High Caloric Food Consumption?", options=["yes", "no"])
with col7:
    fcvc_ui = st.selectbox("Vegetable Consumption Frequency", 
                           options=["Never", "Sometimes", "Always (With every meal)"], index=1)
with col8:
    ncp_ui = st.slider("Number of Main Meals Daily", min_value=1, max_value=4, value=3, step=1)

caec = st.selectbox("Eating Food Between Meals (Snacking)", options=["No", "Sometimes", "Frequently", "Always"], index=1)

st.write("---")
st.header("🏃‍♂️ Section 3: Lifestyle & Transport")
col9, col10, col11 = st.columns(3)
with col9:
    ch2o_ui = st.selectbox("Daily Water Consumption", 
                           options=["Less than 1 Liter", "1 to 2 Liters", "More than 2 Liters"], index=1)
with col10:
    faf_ui = st.selectbox("Physical Activity Frequency", 
                          options=["0 days (No physical activity)", "1 to 2 days / week", "2 to 4 days / week", "4 or more days / week"], index=1)
with col11:
    tue_ui = st.selectbox("Daily Digital Screen Time (Tech Devices)", 
                          options=["0 to 2 hours / day", "3 to 5 hours / day", "More than 5 hours / day"], index=2)

col12, col13, col14 = st.columns(3)
with col12:
    smoke = st.selectbox("Do you Smoke?", options=["yes", "no"])
with col13:
    scc = st.selectbox("Do you Track Daily Calories?", options=["yes", "no"])
with col14:
    calc = st.selectbox("Alcohol Consumption Frequency", options=["no", "Sometimes", "Frequently", "Always"])

mtrans = st.selectbox("Primary Mode of Transportation", options=["Public_Transportation", "Automobile", "Walking", "Motorbike", "Bike"])

st.write("---")

# 3. Prediction Pipeline logic
if st.button("🚀 Analyze Risk Profile", type="primary"):
    
    # Internal mappings translating logical UI items back to raw model expectations
    fcvc_map = {"Never": 1.0, "Sometimes": 2.0, "Always (With every meal)": 3.0}
    ch2o_map = {"Less than 1 Liter": 1.0, "1 to 2 Liters": 2.0, "More than 2 Liters": 3.0}
    faf_map = {"0 days (No physical activity)": 0.0, "1 to 2 days / week": 1.0, "2 to 4 days / week": 2.0, "4 or more days / week": 3.0}
    tue_map = {"0 to 2 hours / day": 0.0, "3 to 5 hours / day": 1.0, "More than 5 hours / day": 2.0}
    
    # Calculate current BMI for deep insight reporting
    calculated_bmi = (weight / (height ** 2))
    
    # Step A: Structural assembly matching raw feature set
    raw_input_df = pd.DataFrame([{
        'Gender': 1 if gender == "Male" else 0,
        'Age': float(age),
        'Height': float(height),
        'Weight': float(weight),
        'family_history_with_overweight': 1 if family_history == "yes" else 0,
        'FAVC': 1 if favc == "yes" else 0,
        'FCVC': fcvc_map[fcvc_ui],
        'NCP': float(ncp_ui),
        'CAEC': {'No': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[caec],
        'SMOKE': 1 if smoke == "yes" else 0,
        'CH2O': ch2o_map[ch2o_ui],
        'SCC': 1 if scc == "yes" else 0,
        'FAF': faf_map[faf_ui],
        'TUE': tue_map[tue_ui],
        'CALC': {'no': 0, 'Sometimes': 1, 'Frequently': 2, 'Always': 3}[calc]
    }])
    
    # Step B: Feature Engineering - Recreate BMI feature mapping
    raw_input_df['BMI'] = round(calculated_bmi, 2)
    
    # Step C: One-Hot Encoding - Recreate dummy sequence structure
    mtrans_modes = ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']
    for mode in mtrans_modes:
        raw_input_df[f'MTRANS_{mode}'] = 1 if mtrans == mode else 0
        
    try:
        # Step D: Align structure columns precisely
        processed_df = raw_input_df[trained_features]
        
        # Step E: Scale data uniformly using training parameters
        scaled_values = scaler.transform(processed_df)
        processed_df_scaled = pd.DataFrame(scaled_values, columns=trained_features)
        
        # Step F: Run model pipeline inference
        prediction_idx = model.predict(processed_df_scaled)[0]
        prediction_probabilities = model.predict_proba(processed_df_scaled)[0]
        confidence_score = np.max(prediction_probabilities) * 100
        
        # Step G: Decode targets for production display
        target_decoder = {
            0: 'Insufficient Weight', 1: 'Normal Weight',
            2: 'Obesity Type I', 3: 'Obesity Type II', 4: 'Obesity Type III',
            5: 'Overweight Level I', 6: 'Overweight Level II'
        }
        display_label = target_decoder.get(int(prediction_idx), "Unknown Class")
        
        # Behavioral Risk Logic Rules
        risk_points = 0
        max_points = 6
        warnings_list = []
        
        if favc == "yes":
            risk_points += 1
            warnings_list.append("🔴 High-Calorie Intake: Regular consumption of energy-dense foods accelerates metabolic stress.")
        if fcvc_ui == "Never":
            risk_points += 1
            warnings_list.append("🔴 Low Micronutrient Range: Insufficient vegetable intake compromises dietary fiber quotas.")
        if caec in ["Frequently", "Always"]:
            risk_points += 1
            warnings_list.append("🔴 Uncontrolled Snacking Pattern: Eating calorie-dense foods between major meals promotes glycemic spikes.")
        if ch2o_ui == "Less than 1 Liter":
            risk_points += 1
            warnings_list.append("🔴 Suboptimal Hydration: Consuming less than 1L of water daily slows basic metabolic processes.")
        if faf_ui == "0 days (No physical activity)":
            risk_points += 1
            warnings_list.append("🔴 Sedentary Profile: Complete absence of physical activity severely drops energy expenditure coefficients.")
        if tue_ui == "More than 5 hours / day":
            risk_points += 1
            warnings_list.append("🔴 Elevated Screen Time: High daily digital device utilization correlates with prolonged physical stasis.")
            
        hazard_index = (risk_points / max_points) * 100

        # ==========================================
        # 🚀 PRESENTATION LAYER: RENDER REPORT
        # ==========================================
        st.header("📋 Clinical Assessment Report")
        
        # Dynamic Diagnostic Card Border Color Customization
        if "Obesity" in display_label:
            card_border_color = "#ff4b4b"  # Clinical Red
        elif "Overweight" in display_label:
            card_border_color = "#ffa500"  # Pre-Clinical Orange
        elif "Normal" in display_label:
            card_border_color = "#28a745"  # Healthy Green
        else:
            card_border_color = "#17a2b8"  # Teal Info Accent
            
        # Full-width card markup to ensure 100% string visibility at massive scale
        st.markdown(f"""
        <div style="background-color: #f8f9fa; padding: 22px; border-radius: 12px; border-left: 8px solid {card_border_color}; margin-bottom: 25px; box-shadow: 0 1px 3px rgba(0,0,0,0.05);">
            <p style="margin: 0; font-size: 13px; color: #6c757d; text-transform: uppercase; font-weight: 700; letter-spacing: 1.2px;">Model Diagnostic Output</p>
            <h1 style="margin: 5px 0 0 0; font-size: 34px; color: #212529; font-weight: 800; line-height: 1.2;">{display_label}</h1>
        </div>
        """, unsafe_allow_html=True)
        
        # Secondary statistics displayed neatly underneath the primary card
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric(label="Calculated BMI Metric", value=f"{calculated_bmi:.2f} kg/m²")
        with res_col2:
            st.metric(label="Classification Certainty", value=f"{confidence_score:.1f}%")
            
        st.write("---")
        
        # Section 2 Metrics: The Justice Framework
        st.subheader("🩺 Behavioral Trend Analysis")
        
        if hazard_index >= 66:
            st.error(f"⚠️ High Lifestyle Hazard Level: {hazard_index:.0f}%")
        elif hazard_index >= 33:
            st.warning(f"⚠️ Moderate Lifestyle Hazard Level: {hazard_index:.0f}%")
        else:
            st.success(f"✅ Low Lifestyle Hazard Level: {hazard_index:.0f}%")
            
        st.progress(hazard_index / 100.0)
        
        # Custom Context Alerts for "Normal Weight" individuals with hidden risks
        if "Normal" in display_label and hazard_index >= 50:
            st.info("""
            💡 **Data Analytics Insight for Presentation:** This case represents a clinical variance. While the machine learning model classifies this individual 
            as **Normal Weight** due to their current biometric constraints ($\text{BMI} = w/h^2$), the behavioral engine reveals 
            severe dietary and stasis risks. Without immediate lifestyle adjustments, this subject presents high vulnerability 
            to moving into 'Overweight Level I' in subsequent screenings.
            """)
            
        if warnings_list:
            st.markdown("### 🔍 Major Behavioral Risk Factors Detected:")
            for warning in warnings_list:
                st.markdown(warning)
        else:
            st.markdown("🎉 **Excellent Habit Metrics:** Lifestyle behaviors match recommendations for maintaining weight stability over time.")
            
    except Exception as e:
        st.error(f"Inference Mapping Error: {e}")