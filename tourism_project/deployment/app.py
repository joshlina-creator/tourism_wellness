"""
Streamlit App - Wellness Tourism Package Purchase Predictor
-------------------------------------------------------------
Loads the trained pipeline (preprocessing + classifier) and lets a sales /
marketing user enter a prospective customer's details to predict whether
they are likely to purchase the Wellness Tourism Package.
"""

import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

MODEL_LOCAL_PATH = "best_model.joblib"
HF_MODEL_REPO = os.environ.get(
    "HF_MODEL_REPO", "Joshlina153/tourism-wellness-package-model"
)


@st.cache_resource
def load_model():
    if os.path.exists(MODEL_LOCAL_PATH):
        return joblib.load(MODEL_LOCAL_PATH)
    model_path = hf_hub_download(repo_id=HF_MODEL_REPO, filename="best_model.joblib")
    return joblib.load(model_path)


st.set_page_config(page_title="Wellness Package Purchase Predictor", page_icon="\U0001F9D8")
st.title("\U0001F9D8 Wellness Tourism Package - Purchase Predictor")
st.write(
    "Enter a prospective customer's details below to predict whether they are "
    "likely to purchase the newly launched **Wellness Tourism Package**."
)

model = load_model()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    typeof_contact = st.selectbox("Type of Contact", ["Self Enquiry", "Company Invited"])
    city_tier = st.selectbox("City Tier", [1, 2, 3])
    occupation = st.selectbox("Occupation", ["Salaried", "Free Lancer", "Small Business", "Large Business"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    num_persons = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
    num_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=3)
    product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])
    duration_of_pitch = st.number_input("Duration of Pitch (minutes)", min_value=1, max_value=180, value=15)
    preferred_star = st.selectbox("Preferred Property Star", [3.0, 4.0, 5.0])

with col2:
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    num_trips = st.number_input("Average Number of Trips per Year", min_value=0, max_value=25, value=3)
    passport = st.selectbox("Holds a Passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a Car?", ["Yes", "No"])
    num_children = st.number_input("Number of Children Visiting (below age 5)", min_value=0, max_value=5, value=0)
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
    monthly_income = st.number_input("Monthly Income", min_value=1000, max_value=200000, value=22000)
    pitch_satisfaction = st.selectbox("Pitch Satisfaction Score", [1, 2, 3, 4, 5])

if st.button("Predict Purchase Likelihood", type="primary"):
    input_df = pd.DataFrame([{
        "Age": age, "TypeofContact": typeof_contact, "CityTier": city_tier,
        "DurationOfPitch": duration_of_pitch, "Occupation": occupation, "Gender": gender,
        "NumberOfPersonVisiting": num_persons, "NumberOfFollowups": num_followups,
        "ProductPitched": product_pitched, "PreferredPropertyStar": preferred_star,
        "MaritalStatus": marital_status, "NumberOfTrips": num_trips,
        "Passport": 1 if passport == "Yes" else 0, "PitchSatisfactionScore": pitch_satisfaction,
        "OwnCar": 1 if own_car == "Yes" else 0, "NumberOfChildrenVisiting": num_children,
        "Designation": designation, "MonthlyIncome": monthly_income,
    }])

    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    st.divider()
    if prediction == 1:
        st.success(f"Likely to purchase the Wellness Package (confidence: {probability:.1%})")
    else:
        st.warning(f"Unlikely to purchase the Wellness Package (confidence: {1 - probability:.1%})")
