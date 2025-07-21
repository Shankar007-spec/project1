import streamlit as st
import joblib
import pandas as pd
import plotly.express as px
import time
from datetime import datetime


# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="💼 Salary Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------- STYLING ----------
st.markdown("""<style>
    body { font-family: 'Inter', sans-serif; }
    .main-header { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    padding: 2rem; border-radius: 15px; margin-bottom: 2rem;
    box-shadow: 0 10px 30px rgba(0,0,0,0.1); }
    .main-header h1 { color: white; text-align: center; font-weight: 700; font-size: 2.5rem; margin: 0; }
    .prediction-card { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
    padding: 2rem; border-radius: 15px; text-align: center; }
    .prediction-amount { font-size: 2.5rem; font-weight: 700; color: white; margin: 1rem 0; }
</style>""", unsafe_allow_html=True)


# ---------- MODEL ----------
@st.cache_resource
def load_model():
    return joblib.load("salary_prediction_pipeline.pkl")


# ---------- VALIDATION ----------
def validate_age(age):
    return (18 <= age <= 60), "Age must be between 18 and 100"


def validate_experience(experience, age):
    if experience < 0:
        return False, "Experience cannot be negative"
    if experience > (age - 16):
        return False, "Experience cannot exceed (Age - 16) years"
    return True, ""


# ---------- MAIN FUNCTION ----------
def main():
    model = load_model()

    # ----- HEADER -----
    st.markdown("""<div class="main-header"><h1>💼Employee Salary Predictor</h1></div>""", unsafe_allow_html=True)

    # ----- PREDICTION SETTINGS AT TOP -----
    st.subheader("Prediction Settings")
    settings_col1, settings_col2 = st.columns(2)
    with settings_col1:
        show_charts = st.checkbox("Show Visualization", value=True)
    with settings_col2:
        currency = st.selectbox("Currency", ["₹ (INR)", "$ (USD)", "€ (EUR)"])

    # ----- USER INPUTS -----
    col1, col2 = st.columns([2, 1])
    with col1:
        age = st.number_input("Age", min_value=18, max_value=100, value=30)
        gender = st.selectbox("Gender", ("Male", "Female"))
        education_level = st.selectbox("Education Level", ["High School", "Bachelor", "Master", "PhD"], index=1)
        job_title = st.selectbox("Job Title", ["Developer", "Data Scientist", "Manager", "Analyst", "Engineer"])
        experience = st.slider("Years of Experience", min_value=0, max_value=50, value=5)

    with col2:
        age_valid, age_msg = validate_age(age)
        exp_valid, exp_msg = validate_experience(experience, age)
        if not age_valid:
            st.error(age_msg)
        if not exp_valid:
            st.error(exp_msg)

    st.markdown("---")

    # ----- PREDICT -----
    if st.button("🔮 Predict My Salary", use_container_width=True):
        if not (age_valid and exp_valid):
            st.error("❌ Please fix the validation errors above before predicting.")
        else:
            with st.spinner("Analyzing your profile..."):
                time.sleep(1)

                input_df = pd.DataFrame({
                    "Age": [age],
                    "Gender": [gender],
                    "Education Level": [education_level],
                    "Job Title": [job_title],
                    "Years of Experience": [experience]
                })

                prediction = model.predict(input_df)[0]

                currency_multipliers = {
                    "₹ (INR)": 1,
                    "$ (USD)": 0.012,
                    "€ (EUR)": 0.011,
                }
                symbol = currency.split()[0]
                converted_salary = prediction * currency_multipliers[currency]

                st.markdown(f"""
                    <div class="prediction-card"><p>💰 Predicted Annual Salary</p>
                    <div class="prediction-amount">{symbol} {converted_salary:,.2f}</div></div>""",
                    unsafe_allow_html=True)

                col1, col2, col3 = st.columns(3)
                col1.metric("Monthly Salary", f"{symbol} {converted_salary / 12:,.2f}")
                col2.metric("Hourly Rate", f"{symbol} {converted_salary / (40 * 52):.2f}")
                col3.metric("Daily Earning", f"{symbol} {converted_salary / 365:.2f}")

                # ----- VISUALIZATIONS -----
                if show_charts:
                    # SCATTER PLOT: Years of Experience vs Actual Salary
                    st.markdown("### 📊 Actual vs. Predicted Salary (Scatter Plot)")
                    scatter_data = pd.DataFrame({
                        "Job Role": ["Developer", "Data Scientist", "Manager", "Analyst", "Engineer"],
                        "Actual Salary (INR)": [800000, 1200000, 1500000, 700000, 900000],
                        "Years of Experience": [3, 5, 10, 2, 4]
                    })

                    scatter_data.loc[len(scatter_data)] = {
                        "Job Role": "You",
                        "Actual Salary (INR)": prediction,
                        "Years of Experience": experience
                    }

                    fig = px.scatter(
                        scatter_data,
                        x="Years of Experience",
                        y="Actual Salary (INR)",
                        color="Job Role",
                        title="Actual vs Predicted Salary (INR)",
                        size=[10] * len(scatter_data),
                        hover_data=["Job Role"]
                    )

                    fig.update_traces(marker=dict(line=dict(width=2, color='DarkSlateGrey')))
                    st.plotly_chart(fig, use_container_width=True)

                    # PIE CHART: Role Distribution
                    st.markdown("### 📊 Job Role Distribution (Pie Chart)")
                    role_distribution = pd.DataFrame({
                        "Role": ["Developer", "Data Scientist", "Manager", "Analyst", "Engineer"],
                        "Count": [50, 20, 15, 10, 5]
                    })
                    pie_fig = px.pie(
                        role_distribution,
                        names="Role",
                        values="Count",
                        title="Job Role Distribution",
                        color_discrete_sequence=px.colors.sequential.RdBu
                    )
                    st.plotly_chart(pie_fig, use_container_width=True)

                # ----- DOWNLOAD REPORT -----
                result_data = {
                    "Prediction_Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Age": age,
                    "Gender": gender,
                    "Education": education_level,
                    "Job_Title": job_title,
                    "Experience": experience,
                    "Predicted_Salary": prediction,
                    "Currency": currency,
                    "Converted_Salary": converted_salary
                }
                result_df = pd.DataFrame([result_data])
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prediction Report",
                    data=csv,
                    file_name=f"salary_prediction_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                st.success("✅ Prediction completed!")


# ---------- EXECUTE ----------
if __name__ == "__main__":
    main()