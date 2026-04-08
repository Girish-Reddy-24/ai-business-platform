import streamlit as st
import requests

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(
    page_title="AI Business Dashboard",
    layout="wide"
)

st.title("📊 AI Business Decision Platform")

# ===============================
# 🔴 CHURN SECTION
# ===============================
st.subheader("🔴 Customer Churn Analysis")

col1, col2, col3 = st.columns(3)

with col1:
    tenure = st.number_input("Tenure (months)", min_value=0.0)

with col2:
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)

with col3:
    total_charges = st.number_input("Total Charges", min_value=0.0)


if st.button("🚀 Analyze Customer"):

    url = "https://ai-business-intelligence-platform.onrender.com/predict"

    payload = {
        "tenure": tenure,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }

    response = requests.post(url, json=payload)

    if response.status_code == 200:
        data = response.json()

        if "error" in data:
            st.error(data["error"])
        else:
            prediction = data["prediction"]
            probability = data["churn_probability"]
            factors = data["top_factors"]

            # -------------------------------
            # KPI CARDS
            # -------------------------------
            colA, colB = st.columns(2)

            with colA:
                if "leave" in prediction:
                    st.error("⚠️ High Churn Risk")
                else:
                    st.success("✅ Low Churn Risk")

            with colB:
                st.metric("Risk Score", f"{probability}%")

            # -------------------------------
            # PROGRESS BAR
            # -------------------------------
            st.progress(int(probability))

            # -------------------------------
            # TOP FACTORS
            # -------------------------------
            st.subheader("🔍 Key Drivers")

            for f in factors:
                st.markdown(f"👉 **{f}**")

    else:
        st.error("API connection failed")


# ===============================
# 📈 FORECAST SECTION
# ===============================
st.markdown("---")
st.subheader("📈 Demand Forecasting")

days = st.slider("Select forecast duration (days)", 1, 30, 10)

if st.button("📊 Generate Forecast"):

    response = requests.get(f"https://ai-business-intelligence-platform.onrender.com/forecast?days={days}")

    if response.status_code == 200:
        forecast = response.json()["forecast"]

        st.line_chart(forecast)
        st.write("Forecast Values:")
        st.write(forecast)

    else:
        st.error("Forecast API failed")


# ===============================
# 💬 NLP SECTION
# ===============================
st.markdown("---")
st.subheader("💬 Customer Feedback Insights")

if st.button("🔍 Analyze Feedback"):

    response = requests.post("https://ai-business-intelligence-platform.onrender.com/nlp-insights")

    if response.status_code == 200:
        insights = response.json()["insights"]

        for key, value in insights.items():
            st.markdown(f"### {key.upper()}")
            st.write(", ".join(value))

    else:
        st.error("NLP API failed")