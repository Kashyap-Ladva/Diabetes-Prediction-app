import streamlit as st 
import pandas as pd 
import numpy as np 
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns 
import pickle

st.set_page_config(page_title="Diabetes Prediction App",layout="centered")

def file_ok(p: Path) -> bool:
    return p.exists() and p.stat().st_size > 0

def safe_joblib_load(path: Path):
    """
    Load a joblib file with helpful error messages for EOF / Unpickling problems.
    Raises RuntimeError with user-friendly advice on failure.
    """
    if not path.exists():
        raise RuntimeError(f"File not found: {path}. Please run the training script to create it.")
    if path.stat().st_size == 0:
        raise RuntimeError(f"File {path} is empty (0 bytes). It appears corrupted. Recreate it by retraining.")
    try:
        return joblib.load(path)
    except EOFError as e:
        raise RuntimeError(
            f"EOFError while loading {path} — file looks truncated/corrupt. "
            "Try re-generating the model file by re-running training or restoring a backup. "
            f"Original error: {e}"
        )
    except pickle.UnpicklingError as e:
        raise RuntimeError(
            f"UnpicklingError while loading {path} — incompatible or corrupted pickle. "
            "Possible causes: mismatched Python/scikit-learn/joblib versions or corrupted file. "
            f"Original error: {e}"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load {path}: {e}")


model_path = Path("models/diabetes_model.pkl")
scaler_path = Path("models/scaler.pkl")
data_path = Path("data/diabetes.csv")

@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    return model,scaler

@st.cache_data
def load_data():
    return pd.read_csv(data_path)

df = load_data()

if "Outcome" not in df.columns:
    st.error("Dataset doesn't have an 'Outcome' column.")
    st.stop()

feature_names = [c for c in df.columns if c != "Outcome"]

try:
    model, scaler = load_model_and_scaler()
except FileNotFoundError:
    st.error("Model or scaler file not found in `models/` directory. Run training first.")
    st.stop()
except Exception as e:
    st.error(f"Failed to load model/scaler: {e}")
    st.stop()

st.title(" Diabetes Risk Prediction App ")
st.write(
    "This app uses a Logistic Regression model trained on the Pima Indians Diabetes dataset "
   "This demo is for educational purposes only."
)

st.sidebar.header("About")
st.sidebar.markdown(
    """
    **Tech Stack**  
    - Python
    - Pandas, NumPy  
    - Matplotlib, Seaborn  
    - Scikit-learn (Logistic Regression)  
    - Joblib
    - Streamlit  
    
    **Flow**  
    Data → Preprocess → Train ML model → Save →  
    Load in Streamlit → Predict.
    """
)

st.sidebar.markdown("---")
st.sidebar.markdown("Built as a learning project — not medical advice.")

st.subheader("Enter Patient Details")

with st.form("Prediction_Form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age", min_value=1, max_value=120, value=25)
        glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
        blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=80)
        skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)

    with col2:
        insulin = st.number_input("Insulin", min_value=0, max_value=900, value=100)
        bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=20.0 , format="%.1f")
        dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0,value = 0.500, format="%.3f")
        pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=2)
    
    submitted = st.form_submit_button("Predict Diabetes Risk")

input_data = {
    "Pregnancies": pregnancies,
    "Glucose": glucose,
    "BloodPressure": blood_pressure,
    "SkinThickness": skin_thickness,
    "Insulin": insulin,
    "BMI": bmi,
    "DiabetesPedigreeFunction": dpf,
    "Age": age,
}

input_df = pd.DataFrame([input_data])

try:
    input_df = input_df[feature_names]
except Exception:
    # If ordering fails, just use the dataframe as-is but warn
    st.warning("Warning: Could not reorder input features to match training features. Prediction may fail or be incorrect.")

st.markdown("### Prediction ")

if submitted:
    try:
        # Scale using fitted scaler
        scaled = scaler.transform(input_df)
        pred_class = model.predict(scaled)[0]
        pred_proba = model.predict_proba(scaled)[0][1]  # probability of class 1 (diabetes)

        if pred_class == 1:
            st.error(f"High risk of diabetes.\n\nEstimated probability: **{pred_proba:.2%}**")
        else:
            st.success(f"Low risk of diabetes.\n\nEstimated probability: **{pred_proba:.2%}**")

    except Exception as e:
        st.error(f"Prediction failed: {e}")

st.markdown("---")
st.subheader("Dataset Insights")

with st.expander("Show Basic Statistics & Distributions"):
    st.write("Sample of Dataset:")
    st.dataframe(df.head())

    st.write("Basic Statitics:")
    st.dataframe(df.describe())

    
    # Glucose distribution by outcome
    fig1, ax1 = plt.subplots(figsize=(6, 4))
    sns.histplot(data=df, x="Glucose", hue="Outcome", kde=True, ax=ax1, alpha=0.6)
    ax1.set_title("Glucose distribution by Outcome (0 = No, 1 = Yes)")
    st.pyplot(fig1)

    fig2, (ax2, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))
    sns.boxplot(data=df, x="Outcome", y="Age", ax=ax2)
    ax2.set_title("Age distribution by Outcome (Boxplot)")
    
    sns.histplot(data=df, x="Age", hue="Outcome", multiple="stack", bins=20, ax=ax3)
    ax3.set_title("Age distribution by Outcome (Stacked Histogram)")
    plt.tight_layout()
    st.pyplot(fig2)

    fig3, (ax4, ax5) = plt.subplots(nrows=1, ncols=2, figsize=(12, 4))

    sns.boxplot(data=df, x="Outcome", y="Insulin", ax=ax4)
    ax4.set_title("Insulin Levels by Outcome (boxplot)")

    sns.histplot(data=df, x="Insulin", hue="Outcome", multiple="stack", bins=25, ax=ax5)
    ax5.set_title("Insulin distribution by Outcome (stacked histogram)")

    plt.tight_layout()
    st.pyplot(fig3)

    st.write("### Correlation heatmap")
    try:
        corr = df.corr()
        fig4, ax6 = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax6, square=False)
        ax6.set_title("Feature Correlation Matrix")
        st.pyplot(fig4)
    except Exception as e:
        st.warning(f"Could not compute correlation heatmap: {e}")


st.markdown("---")
st.caption("**Disclaimer:** This application is for educational purposes only and is not medical advice.")