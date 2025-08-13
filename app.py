import streamlit as st
import joblib
import pandas as pd
import os

# Set page config
st.set_page_config(
    page_title="COVID-19 Risk Assessment",
    page_icon="🦠",
    layout="centered"
)

# Load the trained model with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("final_model.pkl")
        return model
    except FileNotFoundError:
        st.error("Model file 'final_model.pkl' not found. Please ensure the model file is in the same directory as this app.")
        return None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Title and description
st.title("🏥 COVID-19 Risk Assessment Tool")
st.markdown("""
This tool helps assess COVID-19 risk based on symptoms and health conditions.
**Note:** This is not a medical diagnosis. Please consult a healthcare professional for proper testing and advice.
""")

st.divider()

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Demographics")
    age = st.number_input("Age", min_value=0, max_value=120, value=30, help="Enter your age in years")
    gender = st.selectbox("Gender", ["Male", "Female"], help="Select your gender")
    
    st.subheader("Symptoms")
    fever = st.selectbox("Fever", ["No", "Yes"], help="Body temperature above 38°C (100.4°F)")
    cough = st.selectbox("Cough", ["No", "Yes"], help="Persistent dry or wet cough")
    short_breath = st.selectbox("Shortness of Breath", ["No", "Yes"], help="Difficulty breathing or feeling winded")
    loss_of_taste_smell = st.selectbox("Loss of Taste or Smell", ["No", "Yes"], help="Sudden loss of taste or smell")
    fatigue = st.selectbox("Fatigue", ["No", "Yes"], help="Unusual tiredness or exhaustion")

with col2:
    st.subheader("Additional Symptoms")
    headache = st.selectbox("Headache", ["No", "Yes"], help="Persistent or severe headache")
    sore_throat = st.selectbox("Sore Throat", ["No", "Yes"], help="Pain or irritation in throat")
    nausea = st.selectbox("Nausea", ["No", "Yes"], help="Feeling sick or vomiting")
    chest_pain = st.selectbox("Chest Pain", ["No", "Yes"], help="Pain or pressure in chest")
    
    st.subheader("Pre-existing Conditions")
    diabetes = st.selectbox("Diabetes", ["No", "Yes"], help="Type 1 or Type 2 diabetes")
    hypertension = st.selectbox("Hypertension", ["No", "Yes"], help="High blood pressure")

# Mapping function
yes_no_map = {"Yes": 1, "No": 0}
gender_map = {"Male": 1, "Female": 0}

# Convert inputs to numeric form
input_data = {
    'age': age,
    'gender': gender_map[gender],
    'fever': yes_no_map[fever],
    'cough': yes_no_map[cough],
    'short_breath': yes_no_map[short_breath],
    'loss_of_taste_smell': yes_no_map[loss_of_taste_smell],
    'fatigue': yes_no_map[fatigue],
    'headache': yes_no_map[headache],
    'sore_throat': yes_no_map[sore_throat],
    'nausea': yes_no_map[nausea],
    'chest_pain': yes_no_map[chest_pain],
    'diabetes': yes_no_map[diabetes],
    'hypertension': yes_no_map[hypertension]
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

st.divider()

# Display input summary
with st.expander("Review Your Input"):
    symptoms_count = sum([
        yes_no_map[fever], yes_no_map[cough], yes_no_map[short_breath],
        yes_no_map[loss_of_taste_smell], yes_no_map[fatigue], yes_no_map[headache],
        yes_no_map[sore_throat], yes_no_map[nausea], yes_no_map[chest_pain]
    ])
    conditions_count = sum([yes_no_map[diabetes], yes_no_map[hypertension]])
    
    st.write(f"**Age:** {age} years")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Number of symptoms:** {symptoms_count}/9")
    st.write(f"**Pre-existing conditions:** {conditions_count}/2")

# Prediction button
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    if st.button("🔍 Assess Risk", type="primary", use_container_width=True):
        if model is not None:
            try:
                # Make prediction
                prediction = model.predict(input_df)[0]
                
                # Get prediction probability if available
                try:
                    prediction_proba = model.predict_proba(input_df)[0]
                    confidence = max(prediction_proba) * 100
                except:
                    confidence = None
                
                # Display results
                st.divider()
                
                if prediction == 1:
                    st.error("⚠️ **High Risk Assessment**")
                    st.markdown("""
                    Based on the symptoms and conditions provided, there is an elevated risk of COVID-19.
                    
                    **Recommended Actions:**
                    1. **Isolate immediately** to prevent potential spread
                    2. **Get tested** for COVID-19 as soon as possible
                    3. **Contact your healthcare provider** for medical advice
                    4. **Monitor symptoms** and seek emergency care if they worsen
                    
                    **Emergency symptoms requiring immediate medical attention:**
                    - Trouble breathing
                    - Persistent chest pain or pressure
                    - Confusion or inability to stay awake
                    - Bluish lips or face
                    """)
                else:
                    st.success("✅ **Low Risk Assessment**")
                    st.markdown("""
                    Based on the symptoms and conditions provided, the risk of COVID-19 appears to be low.
                    
                    **Still recommended:**
                    1. **Continue monitoring** your symptoms
                    2. **Practice good hygiene** (wash hands, wear mask in crowded places)
                    3. **Stay home if feeling unwell**
                    4. **Consider testing** if symptoms worsen or you've had exposure
                    """)
                
                if confidence is not None:
                    st.info(f"📊 Model confidence: {confidence:.1f}%")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.info("Please check that your model is compatible with the input features.")
        else:
            st.error("Model not loaded. Cannot make predictions.")

# Footer
st.divider()
st.caption("""
**Disclaimer:** This tool is for educational and informational purposes only. It is not a substitute for professional medical advice, diagnosis, or treatment. 
Always seek the advice of your physician or other qualified health provider with any questions you may have regarding a medical condition.
""")

# Add sidebar with additional information
with st.sidebar:
    st.header("ℹ️ About This Tool")
    st.markdown("""
    This COVID-19 risk assessment tool uses machine learning to evaluate the likelihood of COVID-19 based on:
    - Symptoms
    - Age and gender
    - Pre-existing conditions
    
    **Model Information:**
    - File: `final_model.pkl`
    - Features: 13 input variables
    - Output: Binary classification (High/Low risk)
    """)
    
    st.header("📞 Emergency Contacts")
    st.markdown("""
    **Nigeria CDC Hotline:** 07030942066
    
    **SMS:** 08021283200
    
    **WhatsApp:** 07030942066
    """)
    
    st.header("🔗 Resources")
    st.markdown("""
    - [Nigeria CDC COVID-19](https://covid19.ncdc.gov.ng/)
    - [WHO COVID-19 Information](https://www.who.int/emergencies/diseases/novel-coronavirus-2019)
    """)