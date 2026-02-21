import streamlit as st
import pandas as pd
import pickle


st.set_page_config(
    page_title="Subscription Renewal Predictor",
    layout="centered"
)


model = pickle.load(open("model.pkl", "rb"))
model_columns = pickle.load(open("columns.pkl", "rb"))


st.title("Subscription Renewal Prediction App")
st.write("Predict whether a customer will renew their subscription.")

st.markdown("---")


st.header("Enter Customer Details")

usage_days = st.number_input("Usage Days", min_value=0, max_value=365, value=30)
last_login = st.number_input("Days Since Last Login", min_value=0, max_value=365, value=5)
monthly_fee = st.number_input("Monthly Fee", min_value=0.0, value=499.0)

tenure_months = st.number_input("Tenure (Months)", min_value=1, max_value=60, value=12)
support_tickets = st.number_input("Support Tickets", min_value=0, max_value=20, value=1)
satisfaction_score = st.slider("Satisfaction Score", 1.0, 5.0, 3.5)

contract_type = st.selectbox(
    "Contract Type",
    ["Monthly", "Quarterly", "Annual"]
)

payment_method = st.selectbox(
    "Payment Method",
    ["Credit Card", "Debit Card", "UPI", "Net Banking"]
)

plan_type = st.selectbox(
    "Plan Type",
    ["Basic", "Standard", "Premium"]
)

discount_applied = st.selectbox(
    "Discount Applied",
    [0, 1]
)


input_data = pd.DataFrame({
    "usage_days": [usage_days],
    "last_login": [last_login],
    "monthly_fee": [monthly_fee],
    "tenure_months": [tenure_months],
    "support_tickets": [support_tickets],
    "satisfaction_score": [satisfaction_score],
    "discount_applied": [discount_applied],
    "contract_type": [contract_type],
    "payment_method": [payment_method],
    "plan_type": [plan_type]
})


input_data = pd.get_dummies(input_data)


input_data = input_data.reindex(columns=model_columns, fill_value=0)


if st.button("Predict Renewal"):
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    if prediction == 1:
        st.success(
            f"Customer is likely to RENEW\n\nProbability: **{probability:.2f}**"
        )
    else:
        st.error(
            f"Customer is likely to NOT RENEW\n\nProbability: **{probability:.2f}**"
        )