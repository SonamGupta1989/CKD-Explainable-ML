# An Explainable Machine Learning Framework for Predicting Chronic Kidney Disease

## 1. Project Overview

Chronic Kidney Disease (CKD) is a health condition that affects kidney function. Early identification may help healthcare professionals assess patients and make informed decisions.

This project aims to develop an explainable machine learning framework for predicting Chronic Kidney Disease using patient-related clinical data. It combines machine learning predictions with Explainable AI (XAI) techniques to help users understand the factors influencing a prediction.

The project also includes a Streamlit web application that allows users to enter patient information and view a model prediction with an explanation.

**Disclaimer:** This project is an academic research prototype. It is not intended for clinical diagnosis, treatment decisions, or replacement of professional medical advice.

## 2. Project Objectives

- Develop a machine learning framework for CKD prediction.
- Preprocess clinical data, including missing values and categorical features.
- Compare suitable machine learning algorithms using consistent evaluation methods.
- Explore Explainable AI techniques to interpret model predictions.
- Develop an interactive Streamlit web application.
- Evaluate model performance and document the limitations of the approach.

## 3. Machine Learning Algorithms

The following algorithms are included in the project plan for experimentation:

1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Decision Tree Classifier
5. Support Vector Machine (SVM)
6. K-Nearest Neighbors (KNN)

**Implementation status:** Logistic Regression, Random Forest, and Gradient Boosting were included in the initial model comparison. Random Forest is currently used by the Streamlit application. Decision Tree, SVM, and KNN must be added to and evaluated in the comparison code before their experimental results are reported.

The final model will be selected based on appropriate evaluation results, not on algorithm names alone.

## 4. Dataset

The initial experiments use a CKD dataset named `kidney_disease.csv`, containing 400 records and 26 columns.

The dataset includes clinical attributes such as:

- Age and blood pressure
- Specific gravity and albumin
- Blood glucose, blood urea, and serum creatinine
- Haemoglobin and other blood measurements
- Clinical indicators such as hypertension and diabetes
- CKD classification label

The target variable is processed into two classes:

- `1`: CKD
- `0`: Not CKD

**Dataset availability:** The dataset is currently maintained locally and is not included in this GitHub repository. The dataset's original source, license, and permitted usage should be verified before redistribution.

## 5. Data Preprocessing

The machine learning pipeline uses preprocessing steps appropriate for the input data:

- Median imputation for missing numerical values.
- Most-frequent-value imputation for missing categorical values.
- Standardization of numerical features.
- One-hot encoding of categorical features.
- Conversion of the target classification into binary labels.

Preprocessing is integrated into the machine learning pipeline to help ensure that the same transformations are applied during prediction.

## 6. Explainable Artificial Intelligence (XAI)

Explainability is an important component of this project.

The project uses SHAP (SHapley Additive exPlanations) to investigate how input features contribute to model predictions.

The intended explanations include:

- **Global explanation:** Understanding which features influence predictions across a dataset.
- **Local explanation:** Understanding which features contribute to an individual model prediction.

SHAP explanations describe model behaviour; they do not establish medical causation or prove that a prediction is clinically correct.

## 7. Technology Stack

- **Programming language:** Python
- **Data processing:** Pandas, NumPy
- **Machine learning:** Scikit-learn
- **Explainable AI:** SHAP
- **Visualization:** Matplotlib
- **Web application:** Streamlit
- **Model serialization:** Joblib
- **Development environment:** Google Colab and Visual Studio Code
- **Version control:** Git and GitHub

## 8. Project Structure

```text
CKD-Explainable-ML/
├── app.py
├── ckd_random_forest_pipeline.pkl
├── README.md
├── requirements.txt
├── .gitignore
└── kidney_disease.csv  # Local file; not included in this repository
```

The actual repository contents may change as the project develops. The trained model file contains the preprocessing pipeline and Random Forest classifier.

## 9. Installation and Execution

### Prerequisites

- Python installed on your computer
- Git installed
- The project repository cloned or downloaded
- Access to the required dataset if running any training or evaluation scripts

### Step 1: Clone the repository

```bash
git clone https://github.com/SonamGupta1989/CKD-Explainable-ML.git
cd CKD-Explainable-ML
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Run the application

```bash
streamlit run app.py
```

Streamlit will display a local URL in the terminal. Open that URL in your browser.

## 10. Model Evaluation

The models will be compared using appropriate classification metrics, including:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Where appropriate, additional evaluation may include ROC-AUC, cross-validation, and checks for data leakage or class imbalance.

Performance results will be reported after the complete comparison has been run and verified. High performance on a single dataset or split does not, by itself, demonstrate generalization to new patients.

## 11. Current Project Status

- [x] Initial dataset preprocessing and model experimentation
- [x] Initial comparison of Logistic Regression, Random Forest, and Gradient Boosting
- [x] Random Forest pipeline saved
- [x] SHAP-based explanation explored
- [x] Streamlit application implemented
- [x] Project uploaded to GitHub
- [ ] Add Decision Tree, SVM, and KNN to the model comparison
- [ ] Perform consistent model evaluation and validation
- [ ] Finalize and document verified experimental results
- [ ] Review dataset provenance and usage permissions

## 12. Limitations

- The experiments use a limited dataset, and further validation is necessary.
- Performance may vary with different data splits and patient populations.
- Model predictions and SHAP explanations are not substitutes for clinical expertise.
- External validation is needed before any consideration of real-world clinical use.
- Results should not be interpreted as proof of diagnostic accuracy in a healthcare setting.

## 13. Future Work

- Complete the comparison of all six planned algorithms.
- Improve validation and investigate possible data leakage.
- Evaluate feature importance and individual prediction explanations.
- Assess model robustness and generalization.
- Improve the web interface and documentation.
- Explore external validation using suitable, authorized datasets.

## 14. Academic Use

This project is being developed as part of an MTech Computer Science and Engineering dissertation on explainable machine learning for Chronic Kidney Disease prediction.

Please verify dataset attribution, licensing, and any applicable academic requirements before distributing the dataset or publishing research results.
