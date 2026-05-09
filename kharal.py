import streamlit as st
import pandas as pd
import joblib

# Load model + preprocessors
bundle = joblib.load("fertilizer_model.pkl")
clf = bundle["model"]
scaler = bundle["scaler"]
le_crop = bundle["le_crop"]
le_soil = bundle["le_soil"]
le_fert = bundle["le_fert"]
label_mapping = bundle["label_mapping"]
numeric_cols = bundle["numeric_cols"]

st.title("🌱 Fertilizer Recommendation")

# --- User input ---
Temperature = st.number_input("Temperature", value=25.0)
Moisture = st.number_input("Moisture", value=60.0)
Rainfall = st.number_input("Rainfall", value=120.0)
PH = st.number_input("pH", value=6.5)
Carbon = st.number_input("Carbon", value=1.5)
Potassium = st.number_input("Potassium", value=40.0)
Nitrogen = st.number_input("Nitrogen", value=20.0)
Phosphorous = st.number_input("Phosphorous", value=15.0)

Crop = st.selectbox("Crop", le_crop.classes_)
Soil = st.selectbox("Soil", le_soil.classes_)

if st.button("Recommend Fertilizer"):
    # Build input DataFrame
    user_df = pd.DataFrame([{
        "Temperature": Temperature,
        "Moisture": Moisture,
        "Rainfall": Rainfall,
        "PH": PH,
        "Carbon": Carbon,
        "Potassium": Potassium,
        "Nitrogen": Nitrogen,
        "Phosphorous": Phosphorous,
        "Crop": Crop,
        "Soil": Soil,
    }])
    
    # Preprocess
    user_df[numeric_cols] = scaler.transform(user_df[numeric_cols])
    user_df["Crop_le"] = le_crop.transform(user_df["Crop"].astype(str))
    user_df["Soil_le"] = le_soil.transform(user_df["Soil"].astype(str))

    # Predict
    y_pred = clf.predict(user_df[numeric_cols + ["Crop_le", "Soil_le"]])
    fert = le_fert.inverse_transform(y_pred)[0]

    st.success(f"Recommended Fertilizer: **{fert}**")
