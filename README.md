# 🤖 AI Resume Screening & AI Score Prediction

## Developed By
**Kurmapu Lakshmi Tejasree**

---

## Project Overview

This project predicts the AI Score of a candidate based on resume-related features using Machine Learning algorithms.

The objective is to assist recruiters in evaluating candidate profiles efficiently by generating an AI-based score.

---

## Features Used

- Experience (Years)
- Projects Count
- Education
- Certifications
- Job Role
- Skills Count

---

## Machine Learning Models Used

- Linear Regression
- KNN Regressor
- Support Vector Regression (SVR)
- Decision Tree Regressor
- Random Forest Regressor
- XGBoost Regressor

---

## Model Performance

| Model | MAE | RMSE | R² Score |
|---------|---------|---------|---------|
| Linear Regression | 8.52 | 10.17 | 0.81 |
| KNN Regressor | 5.18 | 7.85 | 0.88 |
| SVR | 7.41 | 11.05 | 0.77 |
| Decision Tree | 2.02 | 4.29 | 0.97 |
| Random Forest | 1.70 | 2.88 | 0.98 |
| XGBoost | 1.04 | 1.75 | 0.99 |

---

## Best Model

🏆 **XGBoost Regressor**

- MAE: 1.04
- RMSE: 1.75
- R² Score: 0.99

---

## Feature Importance

The most important features influencing AI Score prediction:

1. Experience (Years)
2. Projects Count
3. Certifications
4. Skills Count
5. Education
6. Job Role

---

## Streamlit Application

The application contains:

### Home Page
- Project Overview
- Features Used
- Models Used

### Dashboard Page
- Model Comparison
- R² Score Visualization
- Feature Importance Analysis

### Prediction Page
- Candidate Input Form
- AI Score Prediction
- Model Comparison Chart

---

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-Learn
- XGBoost
- Matplotlib
- Streamlit
- Joblib


## requirements.txt

- streamlit
- pandas
- numpy
- scikit-learn
- matplotlib
- xgboost
- joblib

## conclusion

This project successfully predicts a candidate's AI Score using Machine Learning techniques. After preprocessing the data and training multiple regression models, XGBoost achieved the best performance with an R² Score of 0.99. The analysis showed that Experience and Projects Count are the most important factors influencing the AI Score. Finally, the model was deployed using Streamlit, allowing users to enter candidate details and instantly receive AI Score predictions.


## Author

**Kurmapu Lakshmi Tejasree**
