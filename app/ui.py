"""
ui.py — Streamlit front-end for ClermontAI credit risk scoring.

All inputs above 1% feature importance are surfaced as primary fields.
Connects to the FastAPI backend at http://localhost:8000.

Run with:
    cd C:\\clermontAI
    venv_new\\Scripts\\python.exe -m streamlit run app/ui.py
"""

import os

import requests
import pandas as pd
import streamlit as st

# Same container -> localhost. Override via env for split deployments.
API_URL = os.environ.get("CLERMONT_API_URL", "http://localhost:8000")

AREA_TYPE_MAP = {
    "Major city centre": 0.0725,  # p99 of training data (actual max)
    "City suburb":       0.0358,  # p90 — also the single most common value in dataset
    "Large town":        0.0189,  # p50 (median)
    "Small town":        0.0100,  # p25
    "Rural / village":   0.0050,  # p5
}

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="ClermontAI — Credit Risk Scorer",
    page_icon="🏦",
    layout="wide",
)

st.title("🏦 ClermontAI Credit Risk Scorer")
st.caption("LightGBM + Platt calibration · Home Credit Default Risk · ROC-AUC ≈ 0.77")

# ---------------------------------------------------------------------------
# Sidebar — API status
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("API Status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=3)
        if r.ok:
            st.success("API connected")
            meta = requests.get(f"{API_URL}/contract", timeout=3).json()
            st.metric("Model version", meta.get("version", "—"))
            st.metric("ROC-AUC (val)", meta.get("validation_roc_auc", "—"))
            st.metric("Features", meta.get("feature_count", "—"))
        else:
            st.error("API returned error")
    except Exception:
        st.error("Cannot reach API at localhost:8000")
        st.info("Start the API first:\n```\ncd C:\\clermontAI\nvenv_new\\Scripts\\python.exe -m uvicorn app.api:app --reload\n```")

    st.divider()
    st.markdown("**Risk tiers**")
    st.markdown("🟢 **LOW** < 15% → Approve")
    st.markdown("🟡 **MEDIUM** 15–40% → Review")
    st.markdown("🔴 **HIGH** > 40% → Decline")

# ---------------------------------------------------------------------------
# Form — wrapping all inputs stops Streamlit rerunning on every widget change
# ---------------------------------------------------------------------------
with st.form("applicant_form"):

    # Section 1 — External Credit Scores  (ranks 1, 2, 4 — 42% of model gain)
    st.subheader("External Credit Bureau Scores")
    st.caption("These three scores are the strongest predictors — together they drive 42% of model decisions. Scale is 0 (highest risk) to 1 (lowest risk).")

    es_col1, es_col2, es_col3 = st.columns(3)
    with es_col1:
        ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.500, 0.001, format="%.3f", help="Bureau score 1 — 5.8% of model gain")
        has_ext1 = st.checkbox("EXT_SOURCE_1 available?", value=False,
                               help="Check if bureau score 1 is on file — leave unchecked to let model impute")
    with es_col2:
        ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.420, 0.001, format="%.3f", help="Bureau score 2 — 17% of model gain")
        has_ext2 = st.checkbox("EXT_SOURCE_2 available?", value=True,
                               help="Check if bureau score 2 is on file — leave unchecked to let model impute")
    with es_col3:
        ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.310, 0.001, format="%.3f", help="Bureau score 3 — 19% of model gain")
        has_ext3 = st.checkbox("EXT_SOURCE_3 available?", value=True,
                               help="Check if bureau score 3 is on file — leave unchecked to let model impute")

    st.divider()

    # Section 2 — Financial  (ranks 8, 13, 19 + drives engineered ranks 3, 7, 9, 17)
    st.subheader("Financial Details")
    st.caption("These four raw values also auto-generate the engineered ratios the model relies on (annuity/credit, credit/goods, annuity/income).")

    fin_col1, fin_col2, fin_col3, fin_col4 = st.columns(4)
    with fin_col1:
        income = st.number_input("Annual income", min_value=0.0, value=135_000.0, step=5_000.0,
                                 help="AMT_INCOME_TOTAL — used to compute debt service burden")
    with fin_col2:
        credit = st.number_input("Loan amount requested", min_value=0.0, value=512_000.0, step=10_000.0,
                                 help="AMT_CREDIT")
    with fin_col3:
        annuity = st.number_input("Annual repayment (annuity)", min_value=0.0, value=24_700.0, step=500.0,
                                  help="AMT_ANNUITY — rank 13 directly, also drives ranks 3 & 17")
    with fin_col4:
        goods_price = st.number_input("Goods / property price", min_value=0.0, value=450_000.0, step=10_000.0,
                                      help="AMT_GOODS_PRICE — rank 8 directly, also drives ranks 7 & 9")

    contract_type = st.selectbox("Loan contract type", ["Cash loans", "Revolving loans"])

    st.divider()

    # Section 3 — Stability & Time Signals  (ranks 5, 6, 14, 15, 18)
    st.subheader("Stability & Time Signals")
    st.caption("How long people have been employed, how recently they moved or changed documents — all are behavioural risk signals.")

    stab_col1, stab_col2 = st.columns(2)

    with stab_col1:
        age_years = st.number_input("Age (years)", min_value=18, max_value=80, value=41,
                                    help="DAYS_BIRTH — rank 6, 3.1% gain")
        employed_years = st.number_input("Years at current job (0 if never employed)",
                                         min_value=0.0, max_value=50.0, value=5.0, step=0.5,
                                         help="DAYS_EMPLOYED — rank 5, 3.6% gain. Enter 0 to flag as never employed.")
        address_years = st.number_input(
            "Years at current address",
            min_value=0.0, max_value=60.0, value=3.0, step=0.5,
            help="DAYS_REGISTRATION — rank 18, 1.35% gain. Residential stability: longer = lower risk."
        )

    with stab_col2:
        id_years = st.number_input(
            "Years since current ID was issued",
            min_value=0.0, max_value=40.0, value=5.0, step=0.5,
            help="DAYS_ID_PUBLISH — rank 15, 1.77% gain. A very recently issued ID can signal fabricated identity."
        )
        phone_change_unit = st.selectbox(
            "How long ago did you last change your phone number?",
            ["Less than 1 month ago", "1–6 months ago",
             "6–12 months ago", "1–3 years ago", "More than 3 years ago"],
            index=2,
            help="DAYS_LAST_PHONE_CHANGE — rank 14, 1.78% gain. Recent changes flag instability."
        )

    st.divider()

    # Section 4 — Employment & Demographics  (ranks 10, 11, 12, 16)
    st.subheader("Employment & Demographics")

    emp_col1, emp_col2, emp_col3 = st.columns(3)

    with emp_col1:
        organization_type = st.selectbox(
            "Employer sector",
            [
                "N/A — not specified", "Business Entity Type 3", "Business Entity Type 2", "Business Entity Type 1",
                "Self-employed", "Government", "Military", "School", "Medicine",
                "Transport: type 4", "Transport: type 3", "Transport: type 2", "Transport: type 1",
                "Construction", "Industry: type 9", "Industry: type 3", "Industry: type 11",
                "Industry: type 7", "Industry: type 1", "Trade: type 7", "Trade: type 3",
                "Trade: type 6", "Trade: type 2", "Police", "Hotel", "Bank", "Insurance",
                "Agriculture", "Postal", "Restaurant", "Security", "Electricity",
                "Telecom", "Emergency", "Cleaning", "Religion", "Other",
            ],
            help="ORGANIZATION_TYPE — rank 10, 2.0% gain. Real-world missingness here is ~0%, so this is mostly for completeness."
        )

    with emp_col2:
        occupation_type = st.selectbox(
            "Occupation",
            [
                "N/A — not specified", "Laborers", "Core staff", "Accountants", "Managers", "Drivers",
                "Sales staff", "Cleaning staff", "Cooking staff", "Private service staff",
                "Medicine staff", "Security staff", "High skill tech staff",
                "Waiters/barmen staff", "Low-skill Laborers", "Realty agents",
                "Secretaries", "IT staff", "HR staff",
            ],
            help="OCCUPATION_TYPE — rank 11, 1.95% gain. Genuinely missing for 31% of real applicants — pick N/A to let the model impute/encode it as unknown, same as a real blank application."
        )

    with emp_col3:
        education_type = st.selectbox(
            "Highest education level",
            [
                "N/A — not specified", "Secondary / secondary special", "Higher education",
                "Incomplete higher", "Lower secondary", "Academic degree",
            ],
            help="NAME_EDUCATION_TYPE — rank 12, 1.84% gain. Real-world missingness here is ~0%, so this is mostly for completeness."
        )
        gender = st.selectbox("Gender", ["M", "F"], help="CODE_GENDER — rank 16, 1.53% gain")

    st.divider()

    # Section 5 — Location & Property  (rank 20)
    st.subheader("Location & Property")

    loc_col1, loc_col2, loc_col3 = st.columns(3)

    with loc_col1:
        area_type = st.selectbox(
            "Area of residence",
            list(AREA_TYPE_MAP.keys()),
            index=1,
            help="REGION_POPULATION_RELATIVE — rank 20, 1.01% gain."
        )

    with loc_col2:
        own_car = st.selectbox("Owns a car?", ["N", "Y"])
        own_car_age = st.number_input("Age of car (years, ignored if no car)",
                                      min_value=0.0, max_value=60.0, value=5.0, step=1.0,
                                      help="OWN_CAR_AGE — rank 21, 0.94% gain")

    with loc_col3:
        family_status = st.selectbox(
            "Family status",
            ["N/A — not specified", "Single / not married", "Married", "Civil marriage", "Widow", "Separated"],
            help="NAME_FAMILY_STATUS. Real-world missingness here is ~0%, so this is mostly for completeness."
        )
        family_members = st.number_input("Total family members", 1, 20, 2)

    st.divider()
    submitted = st.form_submit_button("Score Applicant", type="primary", use_container_width=True)

# ---------------------------------------------------------------------------
# Scoring — only runs when form is submitted
# ---------------------------------------------------------------------------
if submitted:
    phone_change_days_map = {
        "Less than 1 month ago": -15,
        "1–6 months ago":        -90,
        "6–12 months ago":       -270,
        "1–3 years ago":         -730,
        "More than 3 years ago": -1460,
    }

    payload = {
        "NAME_CONTRACT_TYPE":         contract_type,
        "CODE_GENDER":                gender,
        "FLAG_OWN_CAR":               own_car,
        "AMT_INCOME_TOTAL":           income,
        "AMT_CREDIT":                 credit,
        "AMT_ANNUITY":                annuity,
        "AMT_GOODS_PRICE":            goods_price,
        "DAYS_BIRTH":                 -int(age_years * 365.25),
        "DAYS_EMPLOYED":              365243 if employed_years == 0 else -int(employed_years * 365.25),
        "DAYS_REGISTRATION":          -int(address_years * 365.25),
        "DAYS_ID_PUBLISH":            -int(id_years * 365.25),
        "DAYS_LAST_PHONE_CHANGE":     float(phone_change_days_map[phone_change_unit]),
        "REGION_POPULATION_RELATIVE": AREA_TYPE_MAP[area_type],
        "CNT_FAM_MEMBERS":            float(family_members),
    }
    if has_ext1:
        payload["EXT_SOURCE_1"] = ext1
    if has_ext2:
        payload["EXT_SOURCE_2"] = ext2
    if has_ext3:
        payload["EXT_SOURCE_3"] = ext3
    if own_car == "Y":
        payload["OWN_CAR_AGE"] = own_car_age
    NA = "N/A — not specified"
    if organization_type != NA:
        payload["ORGANIZATION_TYPE"] = organization_type
    if occupation_type != NA:
        payload["OCCUPATION_TYPE"] = occupation_type
    if education_type != NA:
        payload["NAME_EDUCATION_TYPE"] = education_type
    if family_status != NA:
        payload["NAME_FAMILY_STATUS"] = family_status

    try:
        resp = requests.post(f"{API_URL}/explain", json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()

        prob        = result["default_probability"]
        tier        = result["risk_tier"]
        decision    = result["decision"]
        base_prob   = result["base_probability"]
        top_features = result["top_features"]

        # --- Score summary ---
        st.subheader("Scoring Result")
        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Default Probability", f"{prob:.1%}")
        r2.metric("Risk Tier", tier)
        r3.metric("Decision", decision)
        r4.metric("Model Baseline", f"{base_prob:.1%}",
                  help="Actual default rate in training data (8.07%)")

        if tier == "LOW":
            st.success(f"✅ **APPROVE** — Default probability {prob:.1%} is below the 15% threshold.")
        elif tier == "MEDIUM":
            st.warning(f"⚠️ **MANUAL REVIEW** — Default probability {prob:.1%} falls in the 15–40% range.")
        else:
            st.error(f"❌ **DECLINE** — Default probability {prob:.1%} exceeds the 40% threshold.")

        st.progress(min(prob, 1.0))
        st.caption(f"0% (no risk) ←————————————→ 100% (certain default)  |  Score: {prob:.1%}")

        # --- SHAP chart ---
        st.divider()
        st.subheader("Top 10 Features Driving This Decision")
        st.caption(
            "Values are in **probability units** — e.g. +0.05 means this feature added 5 pp of default risk. "
            "Red = pushes toward default · Blue = pushes toward repayment."
        )

        df_shap = pd.DataFrame(top_features).sort_values("shap_value", ascending=True)

        chart_data = pd.DataFrame({
            "Feature":            df_shap["feature"].apply(lambda x: x.replace("_", " ")),
            "SHAP contribution":  df_shap["shap_value"],
            "Feature value":      df_shap["raw_value"].round(3),
        })

        try:
            import altair as alt
            chart = (
                alt.Chart(chart_data)
                .mark_bar()
                .encode(
                    x=alt.X("SHAP contribution:Q",
                            title="SHAP value (probability units)",
                            axis=alt.Axis(format=".1%")),
                    y=alt.Y("Feature:N", sort=None, title=""),
                    color=alt.condition(
                        alt.datum["SHAP contribution"] > 0,
                        alt.value("#d62728"),
                        alt.value("#1f77b4"),
                    ),
                    tooltip=["Feature", "SHAP contribution", "Feature value"],
                )
                .properties(height=380)
            )
            st.altair_chart(chart, use_container_width=True)
        except Exception:
            st.dataframe(chart_data, use_container_width=True)

        with st.expander("Raw feature values fed to model"):
            st.dataframe(
                chart_data.rename(columns={"Feature value": "Value fed to model"})
                          .reset_index(drop=True),
                use_container_width=True,
            )

    except requests.exceptions.ConnectionError:
        st.error("Cannot connect to API. Make sure the FastAPI server is running on port 8000.")
    except Exception as e:
        st.error(f"Scoring failed: {e}")
