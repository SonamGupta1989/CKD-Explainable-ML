
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="CKD Explainable ML",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

MODEL_PATH = Path(__file__).parent / "ckd_random_forest_pipeline.pkl"

# ---------------- CUSTOM DESIGN ----------------

st.markdown("""
<style>
.stApp {
    background-color: #f4f7fb;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.main-title {
    font-size: 32px;
    font-weight: 750;
    color: #153451;
}
.subtitle {
    color: #64748b;
    font-size: 16px;
}
.section-title {
    color: #153451;
    font-size: 21px;
    font-weight: 700;
}
div.stButton > button,
div.stFormSubmitButton > button {
    background-color: #176b87;
    color: white;
    border-radius: 9px;
    font-weight: 600;
    border: none;
    min-height: 45px;
}
div.stFormSubmitButton > button:hover {
    background-color: #104e68;
    color: white;
}
[data-testid="stMetric"] {
    background: white;
    padding: 18px;
    border-radius: 12px;
    border: 1px solid #e2e8f0;
}
</style>
""", unsafe_allow_html=True)

# ---------------- MODEL LOADING ----------------

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

st.markdown(
    '<div class="main-title">🩺 Chronic Kidney Disease Prediction</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">An Explainable Machine Learning Framework</div>',
    unsafe_allow_html=True
)

st.write(" ")
st.info(
    "This research application predicts CKD using a trained "
    "Random Forest model and explains predictions using SHAP."
)

st.warning(
    "Academic research prototype only. This prediction is not a "
    "medical diagnosis and must not guide treatment decisions."
)

if not MODEL_PATH.exists():
    st.error("Model file not found. Please check the project folder.")
    st.stop()

try:
    pipeline = load_model()
except Exception as error:
    st.error(f"Model loading failed: {error}")
    st.stop()

# ---------------- SIDEBAR ----------------

with st.sidebar:
    st.title("🩺 CKD Framework")
    st.caption("Explainable Machine Learning")
    st.divider()

    st.markdown("### Project Information")
    st.write("**Model:** Random Forest")
    st.write("**Dataset:** UCI CKD Dataset")
    st.write("**Dataset size:** 400 records")
    st.write("**Features:** 24")
    st.write("**Explainability:** SHAP")

    st.divider()
    st.markdown("### Machine Learning Algorithms")
    st.write("""
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Decision Tree
    - Support Vector Machine
    - K-Nearest Neighbors
    """)

    st.divider()
    st.caption("MTech CSE Dissertation Project")

# ---------------- PATIENT FORM ----------------

st.markdown(
    '<div class="section-title">Enter Patient Details</div>',
    unsafe_allow_html=True
)
st.caption("Enter all 24 patient features to generate a prediction.")

with st.form("patient_form"):

    st.markdown("#### 1. Clinical Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", 1, 120,
            value=None, placeholder="Enter age"
        )
        bp = st.number_input(
            "Blood Pressure (mm/Hg)", 0.0, 300.0,
            value=None, placeholder="Enter BP"
        )
        sg = st.selectbox(
            "Specific Gravity (sg)",
            [1.005, 1.010, 1.015, 1.020, 1.025],
            index=None, placeholder="Select SG"
        )
        al = st.selectbox(
            "Albumin (al)", list(range(6)),
            index=None, placeholder="Select albumin"
        )
        su = st.selectbox(
            "Sugar (su)", list(range(6)),
            index=None, placeholder="Select sugar"
        )
        bgr = st.number_input(
            "Blood Glucose Random (bgr)", 0.0, 2000.0,
            value=None, placeholder="Enter glucose"
        )
        bu = st.number_input(
            "Blood Urea (bu)", 0.0, 2000.0,
            value=None, placeholder="Enter urea"
        )
        sc = st.number_input(
            "Serum Creatinine (sc)", 0.0, 200.0,
            value=None, placeholder="Enter creatinine"
        )

    with col2:
        sod = st.number_input(
            "Sodium (sod)", 0.0, 250.0,
            value=None, placeholder="Enter sodium"
        )
        pot = st.number_input(
            "Potassium (pot)", 0.0, 30.0,
            value=None, placeholder="Enter potassium"
        )
        hemo = st.number_input(
            "Hemoglobin (hemo)", 0.0, 30.0,
            value=None, placeholder="Enter hemoglobin"
        )
        pcv = st.number_input(
            "Packed Cell Volume (pcv)", 0.0, 100.0,
            value=None, placeholder="Enter PCV"
        )
        wc = st.number_input(
            "White Blood Cell Count (wc)", 0.0, 50000.0,
            value=None, placeholder="Enter WBC"
        )
        rc = st.number_input(
            "Red Blood Cell Count (rc)", 0.0, 10.0,
            value=None, placeholder="Enter RBC"
        )

    with col3:
        rbc = st.selectbox(
            "Red Blood Cells (rbc)", ["normal", "abnormal"],
            index=None, placeholder="Select RBC status"
        )
        pc = st.selectbox(
            "Pus Cell (pc)", ["normal", "abnormal"],
            index=None, placeholder="Select pus cell"
        )
        pcc = st.selectbox(
            "Pus Cell Clumps (pcc)", ["notpresent", "present"],
            index=None, placeholder="Select status"
        )
        ba = st.selectbox(
            "Bacteria (ba)", ["notpresent", "present"],
            index=None, placeholder="Select status"
        )
        htn = st.selectbox(
            "Hypertension (htn)", ["no", "yes"],
            index=None, placeholder="Select status"
        )
        dm = st.selectbox(
            "Diabetes Mellitus (dm)", ["no", "yes"],
            index=None, placeholder="Select status"
        )
        cad = st.selectbox(
            "Coronary Artery Disease (cad)", ["no", "yes"],
            index=None, placeholder="Select status"
        )
        appet = st.selectbox(
            "Appetite (appet)", ["good", "poor"],
            index=None, placeholder="Select appetite"
        )
        pe = st.selectbox(
            "Pedal Edema (pe)", ["no", "yes"],
            index=None, placeholder="Select status"
        )
        ane = st.selectbox(
            "Anemia (ane)", ["no", "yes"],
            index=None, placeholder="Select status"
        )

    submitted = st.form_submit_button(
        "🔍 Predict CKD",
        type="primary",
        use_container_width=True
    )

# ---------------- PREDICTION ----------------

if submitted:

    values = [
        age, bp, sg, al, su, bgr, bu, sc, sod, pot,
        hemo, pcv, wc, rc, rbc, pc, pcc, ba,
        htn, dm, cad, appet, pe, ane
    ]

    if any(value is None for value in values):
        st.warning("Please complete all 24 patient fields before prediction.")

    else:
        input_data = pd.DataFrame([{
            "age": age,
            "bp": bp,
            "sg": sg,
            "al": al,
            "su": su,
            "rbc": rbc,
            "pc": pc,
            "pcc": pcc,
            "ba": ba,
            "bgr": bgr,
            "bu": bu,
            "sc": sc,
            "sod": sod,
            "pot": pot,
            "hemo": hemo,
            "pcv": pcv,
            "wc": wc,
            "rc": rc,
            "htn": htn,
            "dm": dm,
            "cad": cad,
            "appet": appet,
            "pe": pe,
            "ane": ane
        }])

        try:
            prediction = int(pipeline.predict(input_data)[0])
            probabilities = pipeline.predict_proba(input_data)[0]

            classifier = pipeline.named_steps["classifier"]
            preprocessor = pipeline.named_steps["preprocessor"]

            class_labels = list(classifier.classes_)
            ckd_index = class_labels.index(1)
            ckd_probability = float(probabilities[ckd_index])

            st.divider()
            st.markdown("## Prediction Results")

            result_col, probability_col = st.columns(2)

            with result_col:
                if prediction == 1:
                    st.error("### CKD Predicted")
                    st.write("The model classified this input as CKD.")
                else:
                    st.success("### Not CKD Predicted")
                    st.write("The model classified this input as Not CKD.")

            with probability_col:
                st.metric(
                    "Estimated CKD Probability",
                    f"{ckd_probability * 100:.2f}%"
                )
                st.progress(ckd_probability)

            st.caption(
                "This is a model-generated probability, not a confirmed "
                "clinical risk or diagnosis."
            )

            # -------- SHAP EXPLANATION --------

            st.divider()
            st.markdown("## Explainable AI – SHAP Analysis")
            st.write(
                "The following graph shows which input features influenced "
                "the model's prediction and in which direction."
            )

            transformed = preprocessor.transform(input_data)

            if hasattr(transformed, "toarray"):
                transformed = transformed.toarray()

            feature_names = preprocessor.get_feature_names_out()

            explainer = shap.TreeExplainer(classifier)
            shap_values = explainer.shap_values(transformed)

            if isinstance(shap_values, list):
                shap_array = np.asarray(shap_values[ckd_index])
                shap_row = shap_array[0]
            else:
                shap_array = np.asarray(shap_values)

                if shap_array.ndim == 3:
                    shap_row = shap_array[0, :, ckd_index]
                elif shap_array.ndim == 2:
                    shap_row = shap_array[0]
                else:
                    raise ValueError("Unexpected SHAP output dimensions.")

            explanation_df = pd.DataFrame({
                "Feature": feature_names,
                "SHAP Value": shap_row
            })

            explanation_df["Absolute Importance"] = (
                explanation_df["SHAP Value"].abs()
            )

            explanation_df = explanation_df.nlargest(
                10, "Absolute Importance"
            ).sort_values("SHAP Value")

            fig, ax = plt.subplots(figsize=(10, 5))

            colors = [
                "tomato" if x > 0 else "steelblue"
                for x in explanation_df["SHAP Value"]
            ]

            ax.barh(
                explanation_df["Feature"],
                explanation_df["SHAP Value"],
                color=colors
            )
            ax.axvline(0, color="black", linewidth=0.8)
            ax.set_xlabel("SHAP value for CKD class")
            ax.set_title("Top 10 Features Influencing the Prediction")
            fig.tight_layout()

            st.pyplot(fig)
            plt.close(fig)

            st.caption(
                "Positive SHAP values push the model output towards CKD; "
                "negative values push it away from CKD. These values explain "
                "model behaviour, not medical causation."
            )

            with st.expander("View Patient Input Details"):
                st.dataframe(input_data, use_container_width=True)

        except Exception as error:
            st.error(f"Prediction or explanation failed: {error}")

# ---------------- ABOUT PROJECT ----------------

st.divider()

with st.expander("About This Research Project"):
    st.write("""
    **Objective:** To develop an explainable machine learning framework
    for predicting Chronic Kidney Disease.

    **Dataset:** UCI Chronic Kidney Disease dataset.

    **Algorithms evaluated:**
    - Logistic Regression
    - Random Forest
    - Gradient Boosting
    - Decision Tree
    - Support Vector Machine
    - K-Nearest Neighbors

    **Selected model:** Random Forest.

    **Explainability:** SHAP-based local feature explanations.

    This academic research prototype requires independent validation
    and must not be used for clinical decisions.
    """)

st.caption(
    "MTech CSE Dissertation | Explainable Machine Learning for CKD Prediction"
)
