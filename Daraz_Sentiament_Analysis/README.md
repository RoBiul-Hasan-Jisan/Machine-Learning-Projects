# Bangla Sentiment Analysis using Bangla-BERT and Focal Loss

## Overview
This project implements a Bangla Sentiment Analysis system for Daraz Bangladesh product reviews using a pretrained Bangla-BERT model.

The system classifies customer reviews into five distinct sentiment levels (1–5 ratings). Because real-world review data is rarely evenly distributed, this project specifically tackles **severe class imbalance** using:
* Weighted Random Sampling
* Focal Loss
* Transformer-based contextual embeddings

The entire pipeline is built and trained using **PyTorch** and **HuggingFace Transformers** on a dataset of Bangladeshi e-commerce reviews.

---

## 1. Dataset

* **Dataset Used:** Daraz Bangladesh Reviews Dataset
* **link :** https://www.kaggle.com/datasets/robiulhasanjisan/bangladesh-daraz-reviews-for-sentiment-analysis
* **Path Used in Training:** `/kaggle/input/datasets/robiulhasanjisan/bangladesh-daraz-reviews-for-sentiment-analysis/Daraz_Master_Reviews_bd.csv`

### Dataset Fields
| Column | Description |
| :--- | :--- |
| `review` | Customer review text in Bangla |
| `rating` | Product rating (1–5) |

### Rating Distribution (The Imbalance Challenge)
The dataset is highly imbalanced, heavily skewed toward 5-star reviews. This imbalance is one of the primary challenges addressed in this project.

| Rating | Count |
| :---: | :--- |
| **5** | 7,229 |
| **4** | 602 |
| **3** | 262 |
| **2** | 123 |
| **1** | 419 |



---

## 2. Model Architecture
The system leverages a pretrained Bangla BERT encoder paired with a custom classification head.

* **Backbone Model:** `sagorsarker/bangla-bert-base`

**Architecture Pipeline:**
```bash
Input Text
   ↓
Tokenizer
   ↓
Bangla-BERT Encoder
   ↓
Pooler Output
   ↓
Linear Classification Layer
   ↓
5-Class Sentiment Prediction
```



---

## 3. Handling Class Imbalance
Standard loss functions fail when one class dominates the dataset. Two distinct techniques are used to mitigate this issue:

### A. Weighted Random Sampling
During the data loading phase, each class is sampled inversely proportional to its frequency in the dataset. 
`weight = 1 / class_frequency`
This ensures that the rare minority classes (like 2-star and 3-star reviews) appear much more often during the training loop.

### B. Focal Loss
Standard Cross Entropy loss treats all errors equally. **Focal Loss** dynamically scales the loss based on prediction confidence, focusing the model's learning efforts on hard-to-classify, minority samples while down-weighting easy, majority samples.

$$FL = (1 - p_t)^\gamma \times CE$$

* $\gamma$ = The focusing parameter (controls the rate at which easy examples are down-weighted).
* $CE$ = Standard Cross Entropy loss.



---

## 4. Training Configuration

| Parameter | Value |
| :--- | :--- |
| **Model** | Bangla-BERT |
| **Epochs** | 3 |
| **Batch Size** | 16 |
| **Learning Rate** | 2e-5 |
| **Max Length** | 128 |
| **Optimizer** | AdamW |
| **Weight Decay** | 0.01 |

**Frameworks Used:** PyTorch, HuggingFace Transformers, Scikit-learn.

---

## 5. Evaluation Metrics
The model is evaluated using standard classification metrics. 
* **Overall Accuracy:** 0.0631
* **Overall Macro F1-score:** 0.0786

### Detailed Classification Report
| Rating | Precision | Recall | F1-Score |
| :---: | :---: | :---: | :---: |
| **1** | 0.1198 | 0.5119 | 0.1941 |
| **2** | 0.0081 | 0.0800 | 0.0148 |
| **3** | 0.0418 | 0.3269 | 0.0741 |
| **4** | 0.0619 | 0.3667 | 0.1059 |
| **5** | 0.7500 | 0.0021 | 0.0041 |

*Note: The exceptionally low overall performance highlights the extreme difficulty of learning from highly imbalanced, noisy real-world review data, even when utilizing focal loss and weighted sampling.*

---

## 6. Installation & Usage

**1. Install required dependencies:**
```bash
pip install transformers datasets scikit-learn torch pandas numpy
```

**2. How to Run:**

- Download the dataset.

- Update the dataset path in the code if necessary.

- Run the training script:

```Bash
python train.py
```
---
## 7. Key Features & Future Improvements
- Key Features
  - Bangla NLP: Utilizing state-of-the-art Transformer architecture for the Bengali language.

  - Custom Loss: Implementation of Focal Loss specifically tailored for text classification.

  - Imbalance Mitigation: Advanced weighted sampling strategy combined with modified loss.

  - HuggingFace Pipeline: End-to-end HuggingFace Trainer integration.

- Future Improvements
  - To boost the heavily constrained performance, the following steps are recommended:

   - Data Augmentation: Over-sampling minority classes using techniques like SMOTE or Back-translation.

   - Class-Balanced Loss: Experimenting with alternative loss functions.

   - Larger Models: Utilizing larger Bangla transformer models (e.g., Bangla-BERT-Large).

    - Ensemble Methods: Combining multiple models to smooth out extreme predictions.
--- 
## 8. Applications
- This robust architecture can be deployed for:

   - E-commerce review analysis

   - Customer satisfaction monitoring

   - Product feedback analysis

   - Social media sentiment analysis

  - Bangla NLP academic research
    
---
## Author

**[Robiul Hasan Jisan](https://robiulhasanjisan.vercel.app/)**
