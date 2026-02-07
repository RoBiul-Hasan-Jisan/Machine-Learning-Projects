#  Spam Email Detection using Machine Learning

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Library](https://img.shields.io/badge/Library-Scikit--Learn-orange)
![Status](https://img.shields.io/badge/Status-Complete-green)

**An end-to-end Natural Language Processing (NLP) project that classifies emails as "Spam" or "Not Spam" using TF-IDF vectorization and Logistic Regression.**

---

##  Project Overview

Spam emails are a major security and productivity issue. This project builds an efficient and interpretable machine learning model to automatically detect spam emails based on their textual content.

### Key Highlights
* **NLP-based text preprocessing**: Cleaning raw text for better analysis.
* **TF-IDF feature extraction**: Utilizing Unigrams and Bigrams.
* **Logistic Regression classifier**: Fast, effective, and interpretable.
* **Model Interpretability**: Identifying specific words that trigger spam detection.
* **Deployment Ready**: Model serialization using Pickle/Joblib.

---

##  Dataset Information

* **Dataset Name**: Spam or Not Spam Dataset
* **Source**: Kaggle

| Feature | Description |
| :--- | :--- |
| `email` | Raw email text content |
| `label` | Target variable: **1 = Spam**, **0 = Not Spam** |

> **Challenges handled:** Missing values, noisy raw text data, and high-dimensional feature spaces.

---

##  Methodology

###  1. Text Preprocessing
The following steps were applied to clean the data:
* Converted text to lowercase.
* Removed punctuation and special characters.
* Removed English stopwords (via NLTK).
* Handled missing values.
* Tokenized text for vectorization.

###  2. Feature Engineering (TF-IDF)
We used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical vectors.

* **N-grams**: (1, 2) — Captures single words and phrases.
* **min_df**: 3 — Ignores extremely rare words.
* **max_df**: 0.9 — Ignores words appearing in >90% of emails.

###  3. Model Used: Logistic Regression
We selected Logistic Regression for its speed and interpretability.
* **Why?** It performs well with sparse vectors and provides probability outputs.
* **Training**: 80/20 Train/Test split with Stratified Sampling.
* **Settings**: `max_iter = 3000`.

---

##  Model Evaluation

We prioritized **Recall** to minimize the risk of missing actual spam emails.

* **Accuracy**: Overall performance.
* **Precision**: Accuracy of positive predictions.
* **Recall**: Coverage of actual positive cases.
* **F1-Score**: Harmonic mean of Precision and Recall.

###  Model Interpretability
Analysis of the model coefficients reveals which words drive the predictions:

* **Spam-indicative words**: `win`, `free`, `click`, `offer`
* **Non-spam words**: `meeting`, `project`, `report`

---

##  How to Run the Project

###  Install Dependencies
```bash
pip install pandas numpy nltk scikit-learn matplotlib seaborn
```
---

## Usage Example
Load the saved model and vectorizer to make predictions:
```bash
import joblib


model = joblib.load('spam_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

def predict_email(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    prob = model.predict_proba(text_vec)[0][1]
    
    label = "Spam" if prediction == 1 else "Not Spam"
    return label, prob

## Test
print(predict_email("Congratulations! You won $1000. Click now!"))
## Output: ('Spam', 0.97)

```

---


## Technologies Used
Language: Python

Libraries: Pandas, NumPy, NLTK, Scikit-learn

Visualization: Matplotlib, Seaborn


# Future Improvements
- Implement LightGBM or XGBoost for potential accuracy gains.
- Add ROC-AUC curve visualization.
- Handle class imbalance using SMOTE.
-  Deploy model using Flask or FastAPI.

 # Author
Robiul Hasan Jisan
