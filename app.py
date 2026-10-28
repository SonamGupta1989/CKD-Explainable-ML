
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib.pyplot as plt
from pathlib import Path

st.set_page_config(
    page_title="CKD Explainable ML",
    page_icon="🩺",
    layout="wide"
)

MODEL_PATH = Path(__file__).parent / "ckd_random_forest_pipeline.pkl"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


st.title("🩺 Chronic Kidney Disease Prediction")
st.subheader("An Explainable Machine Learning Framework")

st.write(
    "Enter the patient measurements below to demonstrate "
    "the trained machine learning model."
)

st.warning(
    "Academic research prototype only. This prediction is not "
    "a medical diagnosis and must not guide treatment decisions."
)

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Please place "
        "ckd_random_forest_pipeline.pkl in the project folder."
    )
    st.stop()

try:
    pipeline = load_model()
except Exception as error:
    st.error(f"Could not load the model: {error}")
    st.stop()


st.markdown("### Enter Patient Details")

with st.form("patient_form"):

    st.markdown("#### 1. Basic and Clinical Measurements")

    col1, col2, col3 = st.columns(3)

    with col1:
        age = st.number_input(
            "Age (years)", min_value=1, max_value=120, value=48
        )
        bp = st.number_input(
            "Blood Pressure (mm/Hg)", min_value=0.0,
            max_value=300.0, value=80.0
        )
        sg = st.selectbox(
            "Specific Gravity (sg)",
            [1.005, 1.010, 1.015, 1.020, 1.025],
            index=2
        )
        al = st.selectbox("Albumin (al)", list(range(6)), index=0)
        su = st.selectbox("Sugar (su)", list(range(6)), index=0)
        bgr = st.number_input(
            "Blood Glucose Random (bgr)", min_value=0.0,
            max_value=2000.0, value=121.0
        )
        bu = st.number_input(
            "Blood Urea (bu)", min_value=0.0,
            max_value=2000.0, value=36.0
        )
        sc = st.number_input(
            "Serum Creatinine (sc)", min_value=0.0,
            max_value=200.0, value=1.2
        )

    with col2:
        sod = st.number_input(
            "Sodium (sod)", min_value=0.0,
            max_value=250.0, value=138.0
        )
        pot = st.number_input(
            "Potassium (pot)", min_value=0.0,
            max_value=30.0, value=4.5
        )
        hemo = st.number_input(
            "Hemoglobin (hemo)", min_value=0.0,
            max_value=30.0, value=13.0
        )
        pcv = st.number_input(
            "Packed Cell Volume (pcv)", min_value=0.0,
            max_value=100.0, value=40.0
        )
        wc = st.number_input(
            "White Blood Cell Count (wc)", min_value=0.0,
            max_value=50000.0, value=8000.0
        )
        rc = st.number_input(
            "Red Blood Cell Count (rc)", min_value=0.0,
            max_value=10.0, value=4.5
        )

    with col3:
        rbc = st.selectbox(
            "Red Blood Cells (rbc)",
            ["normal", "abnormal"]
        )
        pc = st.selectbox(
            "Pus Cell (pc)", ["normal", "abnormal"]
        )
        pcc = st.selectbox(
            "Pus Cell Clumps (pcc)",
            ["notpresent", "present"]
        )
        ba = st.selectbox(
            "Bacteria (ba)", ["notpresent", "present"]
        )
        htn = st.selectbox("Hypertension (htn)", ["no", "yes"])
        dm = st.selectbox("Diabetes Mellitus (dm)", ["no", "yes"])
        cad = st.selectbox("Coronary Artery Disease (cad)", ["no", "yes"])
        appet = st.selectbox("Appetite (appet)", ["good", "poor"])
        pe = st.selectbox("Pedal Edema (pe)", ["no", "yes"])
        ane = st.selectbox("Anemia (ane)", ["no", "yes"])

    submitted = st.form_submit_button(
        "Predict CKD",
        type="primary",
        use_container_width=True
    )


if submitted:

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
        class_labels = list(pipeline.named_steps["model"].classes_)
        ckd_index = class_labels.index(1)
        ckd_probability = float(probabilities[ckd_index])

        st.markdown("---")
        st.markdown("### Prediction Result")

        col1, col2 = st.columns(2)

        with col1:
            if prediction == 1:
                st.error("Model prediction: CKD")
            else:
                st.success("Model prediction: Not CKD")

        with col2:
            st.metric(
                "Estimated CKD probability",
                f"{ckd_probability * 100:.1f}%"
            )

        st.caption(
            "This probability is the model's estimated score, "
            "not a confirmed clinical risk."
        )

        st.markdown("---")
        st.markdown("### SHAP Explanation")

        preprocessor = pipeline.named_steps["preprocessing"]
        classifier = pipeline.named_steps["model"]

        transformed = preprocessor.transform(input_data)
        if hasattr(transformed, "toarray"):
            transformed = transformed.toarray()

        feature_names = preprocessor.get_feature_names_out()

        explainer = shap.TreeExplainer(classifier)
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            values = np.asarray(shap_values[ckd_index])[0]
        elif len(np.asarray(shap_values).shape) == 3:
            values = np.asarray(shap_values)[0, :, ckd_index]
        else:
            values = np.asarray(shap_values)[0]

        explanation_df = pd.DataFrame({
            "Feature": feature_names,
            "SHAP Value": values
        })

        explanation_df["Absolute Importance"] = (
            explanation_df["SHAP Value"].abs()
        )

        explanation_df = explanation_df.sort_values(
            "Absolute Importance", ascending=False
        ).head(10).sort_values("SHAP Value")

        fig, ax = plt.subplots(figsize=(9, 5))

        bar_colors = [
            "tomato" if value > 0 else "steelblue"
            for value in explanation_df["SHAP Value"]
        ]

        ax.barh(
            explanation_df["Feature"],
            explanation_df["SHAP Value"],
            color=bar_colors
        )
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("SHAP value for CKD class")
        ax.set_title("Top Features Influencing This Prediction")
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

        st.caption(
            "Positive SHAP values push the model output toward CKD; "
            "negative values push it away from CKD. These values "
            "explain model behaviour, not medical causation."
        )

        with st.expander("View input details"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as error:
        st.error(f"Prediction or explanation failed: {error}")
        st.info(
            "Share the complete error message so we can troubleshoot it."
        )


st.markdown("---")
st.markdown("### About This Project")
st.write(
    "This academic prototype uses a trained Random Forest pipeline "
    "and SHAP-based explanations. Its results require independent "
    "validation and should not be used for clinical decisions."
)
