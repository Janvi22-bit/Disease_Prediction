import streamlit as st
import numpy as np
import pickle

# -------------------------------------------------
# Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Disease Prediction System",
    page_icon="🩺",
    layout="centered"
)

# -------------------------------------------------
# Background Styling + Visibility Fix
# -------------------------------------------------
st.markdown(
    """
    <style>

    /* Background image */
    .stApp {
        background-image: url("https://image2url.com/r2/default/images/1773137843252-05849a7b-f2fb-450a-b317-61c3ddb3d3f2.png");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    /* STRONG dark overlay */
    .stApp::before {
        content: "";
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background: rgba(0,0,0,0.85);
        z-index: -1;
    }

    /* Title styling */
    h1 {
        color: white !important;
        font-size: 52px !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 6px black;
    }

    /* Sub headers */
    h2, h3 {
        color: white !important;
        font-size: 36px !important;
        font-weight: bold !important;
        text-shadow: 2px 2px 6px black;
    }

    /* All text */
    p, label, div {
        color: white !important;
        font-size: 24px !important;
        font-weight: bold !important;
        text-shadow: 1px 1px 5px black;
    }

    /* Multiselect label */
    .stMultiSelect label {
        font-size: 24px !important;
        font-weight: bold !important;
        color: white !important;
    }

    /* Dropdown text */
    .stMultiSelect div {
        font-size: 20px !important;
        font-weight: bold !important;
    }

    /* Button styling */
    .stButton button {
        font-size: 26px !important;
        font-weight: bold !important;
        padding: 12px 30px;
        border-radius: 12px;
        background-color: #2ecc71;
        color: white;
        border: none;
    }

    .stButton button:hover {
        background-color: #27ae60;
    }

    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Title
# -------------------------------------------------
st.title("🩺 Disease Prediction Helper")

st.write(
"""
👋 Hello! This app helps you **guess a possible disease based on symptoms**.

"""
)

# -------------------------------------------------
# Load Model
# -------------------------------------------------
model = pickle.load(open("disease_model.pkl","rb"))
le = pickle.load(open("label_encoder.pkl","rb"))
symptoms = pickle.load(open("symptoms.pkl","rb"))

# -------------------------------------------------
# Precautions Dictionary
# -------------------------------------------------
precautions = {

"fungal infection":[
"Keep the affected area clean and dry",
"Use antifungal creams",
"Wear loose cotton clothes",
"Avoid sharing towels"
],

"allergy":[
"Avoid allergens like dust or pollen",
"Keep your surroundings clean",
"Take medicines given by doctor",
"See a doctor if allergy is serious"
],

"gerd":[
"Avoid spicy and oily foods",
"Eat smaller meals",
"Do not lie down right after eating",
"Drink less coffee or cola"
],

"diabetes":[
"Check blood sugar regularly",
"Exercise every day",
"Eat healthy foods",
"Take medicines as advised by doctor"
],

"hypertension":[
"Eat less salty food",
"Exercise regularly",
"Stay calm and avoid stress",
"Check blood pressure regularly"
],

"migraine":[
"Avoid bright lights and loud sounds",
"Sleep well",
"Drink enough water",
"Try to reduce stress"
],

"malaria":[
"Use mosquito nets while sleeping",
"Avoid standing water around home",
"Wear full sleeve clothes",
"Take proper medicines"
],

"dengue":[
"Avoid mosquito bites",
"Drink lots of fluids",
"Take enough rest",
"See doctor if fever is high"
],

"typhoid":[
"Drink clean and safe water",
"Eat fresh and clean food",
"Finish doctor prescribed medicines",
"Wash hands before eating"
],

"common cold":[
"Drink warm water or soup",
"Take good rest",
"Use steam inhalation",
"Wash hands often"
],

"pneumonia":[
"Take medicines given by doctor",
"Rest properly",
"Drink plenty of fluids",
"Avoid smoking"
],

"heart attack":[
"Seek medical help immediately",
"Eat healthy foods",
"Exercise regularly",
"Control cholesterol levels"
],

"acne":[
"Keep face clean",
"Avoid oily cosmetics",
"Drink plenty of water",
"Avoid touching pimples"
]

}

# -------------------------------------------------
# Symptom Selection
# -------------------------------------------------
st.subheader("🤒 Select Symptoms You Feel")

selected_symptoms = st.multiselect(
"Choose symptoms from the list below:",
symptoms,
key="symptom_selector"
)

# -------------------------------------------------
# Create Input Vector
# -------------------------------------------------
input_data = np.zeros(len(symptoms))

for symptom in selected_symptoms:
    index = list(symptoms).index(symptom)
    input_data[index] = 1

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Predict Disease", key="predict_btn"):

    prediction = model.predict([input_data])

    disease = le.inverse_transform(prediction)[0].strip().lower()

    st.subheader("🧾 Predicted Disease")
    st.success(disease.title())

    # show precautions
    if disease in precautions:

        st.subheader("💡 What You Should Do (Precautions)")

        for p in precautions[disease]:
            st.write("✔", p)

    else:
        st.warning("Precautions not available for this disease.")

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.write("---")
st.write("⚠ This tool is for learning purposes only. Always consult a doctor for medical advice.")