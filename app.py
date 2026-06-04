import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(
    page_title="AI Resume Screening",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
# CRITICAL: Uncomment these lines once your .pkl files are placed in your project directory
# --------------------------------------------------
# LOAD MODELS
# --------------------------------------------------
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Make sure these lines do NOT have a '#' symbol in front of them:
lr = joblib.load(os.path.join(BASE_DIR, "linear.pkl"))
knn = joblib.load(os.path.join(BASE_DIR, "knn.pkl"))
svr = joblib.load(os.path.join(BASE_DIR, "svr.pkl"))
dt = joblib.load(os.path.join(BASE_DIR, "dt.pkl"))
rf = joblib.load(os.path.join(BASE_DIR, "rf.pkl"))
xgb = joblib.load(os.path.join(BASE_DIR, "xgb.pkl"))

# --------------------------------------------------
# SIDEBAR NAVIGATION
# --------------------------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go To",
    ["Home", "Dashboard", "Prediction"]
)

# ==================================================
# HOME PAGE
# ==================================================
if page == "Home":
    st.title("🤖 AI Resume Screening & AI Score Prediction")
    st.markdown("## Developed By Kurmapu Lakshmi Tejasree")

    # Expanded column distribution [1, 5, 1] opens up the horizontal sides
    col1, col2, col3 = st.columns([1, 9, 1])
    with col2:
        # use_container_width=True allows the image to scale out into the widened column space cleanly
        st.image(
            "https://storage.googleapis.com/kaggle-datasets-images/9561659/14940932/d7dae9b85a861b686b0ffcf183ff8d35/dataset-card.png?t=2026-02-24-07-59-41", 
            use_container_width=True
        )
    
    st.markdown("---")

    st.header("📌 Project Objective")
    st.write("""
    This project predicts the AI Score of a candidate using
    Machine Learning algorithms. Multiple regression models 
    were trained and compared to identify the best performing model.
    """)

    st.header("🛠 Features Used")
    features = pd.DataFrame({
        "Features":[
            "Experience (Years)",
            "Projects Count",
            "Education",
            "Certifications",
            "Job Role",
            "Skills Count"
        ]
    })
    st.table(features)
    st.success("🏆 Best Model: XGBoost Regressor")

# ==================================================
# DASHBOARD PAGE
# ==================================================
elif page == "Dashboard":
    st.title("📊 Model Performance Dashboard")

    metrics = pd.DataFrame({
        'Model':[
            'Linear Regression',
            'KNN',
            'SVR',
            'Decision Tree',
            'Random Forest',
            'XGBoost'
        ],
        'MAE':[8.52, 5.18, 7.41, 2.02, 1.70, 1.04],
        'RMSE':[10.17, 7.85, 11.05, 4.29, 2.88, 1.75],
        'R2 Score':[0.81, 0.88, 0.77, 0.97, 0.98, 0.99]
    })

    st.subheader("Model Performance")
    st.dataframe(metrics, use_container_width=True)

    # Re-arranging Dashboard Plots into your ideal horizontal bar chart style
    st.subheader("R² Score Comparison")
    df_sorted = metrics.sort_values(by='R2 Score', ascending=True)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.barplot(x='R2 Score', y='Model', data=df_sorted, palette='viridis', ax=ax)
    ax.set_title("Model Comparison: R2 Score (Higher is Better)", fontweight='bold')
    ax.set_xlim(0, 1.1)
    for i, v in enumerate(df_sorted['R2 Score']):
        ax.text(v + 0.01, i, f"{v:.2f}", va='center', fontweight='bold')
    st.pyplot(fig)

    st.subheader("XGBoost Feature Importance")
    importance_df = pd.DataFrame({
        "Feature":[
            "Experience (Years)",
            "Projects Count",
            "Education",
            "Certification",
            "Job Role",
            "Skills Count"
        ],
        "Importance":[0.697, 0.241, 0.0015, 0.0318, 0.0015, 0.0271]
    })
    
    st.dataframe(importance_df, use_container_width=True)
    imp_sorted = importance_df.sort_values(by='Importance', ascending=True)

    fig2, ax2 = plt.subplots(figsize=(12, 5))
    ax2.barh(imp_sorted["Feature"], imp_sorted["Importance"], color="#db8a60")
    ax2.set_title("Feature Importance Breakdown", fontweight='bold')
    st.pyplot(fig2)

# ==================================================
# PREDICTION PAGE (FULLY ACTIVE & CORRECTED)
# ==================================================
elif page == "Prediction":
    st.title("🎯 AI Score Prediction")

    experience = st.number_input("Experience (Years)", min_value=0, max_value=20, value=1)
    projects = st.number_input("Projects Count", min_value=0, max_value=20, value=1)
    
    education = st.selectbox("Education", ["bsc", "btech", "mba", "mtech", "phd"])
    certification = st.selectbox("Certification", ["cloud", "data_science", "none"])
    job_role = st.selectbox("Job Role", ["ai researcher", "cybersecurity analyst", "data scientist", "software engineer"])
    skills = st.number_input("Skills Count", min_value=1, max_value=20, value=3)

    if st.button("Predict AI Score"):
        edu_map = {"bsc":1, "btech":2, "mba":3, "mtech":4, "phd":5}
        cert_map = {"cloud":0, "data_science":1, "none":2}
        job_map = {"ai researcher":0, "cybersecurity analyst":1, "data scientist":2, "software engineer":3}

        candidate = pd.DataFrame({
            "Experience (Years)":[experience],
            "Projects Count":[projects],
            "Education_Encoded":[edu_map[education]],
            "Certifications_Encoded":[cert_map[certification]],
            "Job Role_Encoded":[job_map[job_role]],
            "Skills_Count":[skills]
        })

        # --- Active Prediction Pipeline Execution Block ---
        try:
            predictions = {
                "Linear Regression": lr.predict(candidate)[0],
                "KNN": knn.predict(candidate)[0],
                "SVR": svr.predict(candidate)[0],
                "Decision Tree": dt.predict(candidate)[0],
                "Random Forest": rf.predict(candidate)[0],
                "XGBoost": xgb.predict(candidate)[0]
            }

            pred_df = pd.DataFrame(predictions.items(), columns=["Model", "Predicted Score"])
            pred_df_sorted = pred_df.sort_values(by="Predicted Score", ascending=True)

            st.subheader("Prediction Results")
            st.dataframe(pred_df, use_container_width=True)
            
            # Horizontal visual alignment matching your design preference
            fig3, ax3 = plt.subplots(figsize=(12, 5))
            bars = ax3.barh(pred_df_sorted["Model"], pred_df_sorted["Predicted Score"], color="#5b84c4")
            ax3.set_title("Predicted AI Score Comparison across Models", fontweight='bold')
            ax3.set_xlabel("Predicted Score Value")
            
            # Value Annotation placement engine
            for bar in bars:
                width = bar.get_width()
                ax3.text(
                    width + 0.5, 
                    bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', 
                    va='center', 
                    ha='left', 
                    fontweight='bold'
                )

            plt.tight_layout()
            st.pyplot(fig3)

        except NameError:
            st.error("Model Error: The models are not yet loaded.")
            st.info("Please uncomment the model loading code lines (lines 22-27) at the top of your python script file once your pkl models are ready!")