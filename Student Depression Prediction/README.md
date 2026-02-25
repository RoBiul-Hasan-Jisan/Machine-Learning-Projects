#  Student Depression Prediction 



A Machine Learning project designed to predict student depression status by analyzing demographic, academic, and lifestyle factors. This repository features a complete end-to-end ML pipeline from data preprocessing to model evaluation.

---

##  Project Overview
Mental health is a critical challenge in modern education. This project builds a predictive system to:
* **Identify** key contributing factors (Academic Stress, Sleep, Financial Pressure).
* **Develop** high-performance classification models.
* **Evaluate** model reliability using professional metrics (ROC-AUC, F1-Score).
* **Provide** a stable, cross-validated pipeline for research and educational use.

##  Dataset Architecture
The data is sourced from **Kaggle** (`Student Depression Dataset.csv`) and contains **27,000+** records (approx.) with the following feature breakdown:

| Category | Features |
| :--- | :--- |
| **Demographics** | Gender, Age, City |
| **Academic** | CGPA, Study Hours, Academic Stress |
| **Lifestyle** | Sleep Duration, Social Interaction, Dietary Habits |
| **Financial** | Financial Stress |
| **Target** | **Depression (Yes/No)** |

---

##  Technologies & Stack
* **Data Handling:** `Pandas`, `NumPy`
* **Visualization:** `Matplotlib`, `Seaborn`
* **Modeling:** `Scikit-learn` (Logistic Regression, Random Forest)
* **Persistence:** `Joblib` (for model saving and deployment)

---

##  Machine Learning Pipeline

### 1. Data Preprocessing
* **Cleaning:** Removal of duplicates and handling missing values.
* **Encoding:** `OneHotEncoder` for categorical variables to ensure numerical compatibility.
* **Scaling:** `StandardScaler` applied to numerical features for model stability.
* **Automation:** Integrated via `ColumnTransformer` to maintain a clean, leak-free workflow.

### 2. Model Evaluation & Performance
The models were tested on a set of **5,581 samples**. The results indicate an "Excellent" rating for discriminative power (ROC-AUC > 0.90).

#### **Primary Classifier Metrics**
| Metric | Value |
| :--- | :--- |
| **Overall Accuracy** | 84.45% |
| **ROC AUC Score** | **0.9184** |
| **Mean CV ROC AUC** | 0.9119 |

#### **Detailed Classification Report**
| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **0 (No Depression)** | 0.82 | 0.79 | 0.81 | 2313 |
| **1 (Depression)** | **0.86** | **0.88** | **0.87** | **3268** |

---

##  Key Insights
* **High Recall (0.88):** The model is particularly strong at identifying students who *actually* have depression, which is vital for early intervention.
* **Generalization:** The high Cross-Validation score (0.91) proves the model is stable and not overfitting to the training data.
* **Stress Correlation:** Academic and Financial stress scores emerged as the highest-weight features in the decision-making process.

---

## Future Roadmap
[ ] SHAP Integration: Add explainability plots for individual student risk factors.

[ ] Web Dashboard: Deploy the model via Streamlit for interactive use.

[ ] Hyperparameter Tuning: Use GridSearchCV for optimal performance.

 ## Disclaimer
This project is for educational and research purposes only. It is not intended for clinical diagnosis. If you or someone you know is struggling, please seek professional medical help.


