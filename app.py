import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Klasifikasi User Behavior",
    layout="centered"
)

st.title("Klasifikasi User Behavior Class")
st.write(
    "Aplikasi ini menggunakan model Decision Tree hasil training dari Google Colab "
    "untuk memprediksi kelas perilaku pengguna perangkat mobile."
)

# Load model dan fitur hasil training dari Colab
model = joblib.load("model_decision_tree.joblib")
features = joblib.load("features.joblib")

st.subheader("Input Data Pengguna")

input_data = {}

for feature in features:
    input_data[feature] = st.number_input(
        label=feature,
        min_value=0.0,
        value=1.0
    )

if st.button("Prediksi"):
    input_df = pd.DataFrame([input_data])

    prediction = model.predict(input_df)[0]

    st.success(f"Hasil Prediksi: User Behavior Class {prediction}")

    if prediction == 1:
        st.info("Kelas 1 menunjukkan perilaku penggunaan perangkat yang rendah.")
    elif prediction == 2:
        st.info("Kelas 2 menunjukkan perilaku penggunaan perangkat yang cukup rendah.")
    elif prediction == 3:
        st.info("Kelas 3 menunjukkan perilaku penggunaan perangkat sedang.")
    elif prediction == 4:
        st.info("Kelas 4 menunjukkan perilaku penggunaan perangkat cukup tinggi.")
    elif prediction == 5:
        st.info("Kelas 5 menunjukkan perilaku penggunaan perangkat tinggi.")

st.subheader("Fitur yang Digunakan Model")
st.write(features)