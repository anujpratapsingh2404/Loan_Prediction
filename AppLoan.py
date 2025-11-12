import streamlit as st
import pickle
import pandas as pd


model = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="🏦 Loan Prediction App", layout="centered")

st.title("🏦 Loan Approval Prediction")
st.write("Fill in the applicant details to check loan approval status.")


with st.form("loan_form"):
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])

    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0, step=100)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0, step=100)
        loan_amount = st.number_input("Loan Amount", min_value=0, step=100)
        loan_amount_term = st.number_input("Loan Amount Term (in months)", min_value=0, step=1)
        credit_history = st.selectbox("Credit History", [1, 0])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

    submit = st.form_submit_button("🔮 Predict Loan Status")


if submit:
    input_df = pd.DataFrame([{
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_amount_term,
        "Credit_History": credit_history,
        "Property_Area": property_area
    }])

    input_df = pd.get_dummies(input_df)


    expected_cols = model.feature_names_in_
    input_df = input_df.reindex(columns=expected_cols, fill_value=0)

    prediction = model.predict(input_df)[0]

    if prediction == 1:
        st.success("✅ Loan Approved!")
    else:
        st.error("❌ Loan Rejected.")
