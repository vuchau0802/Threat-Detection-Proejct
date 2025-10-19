# Threat Detection using Machine Learning
Detecting and classifying harmful online content using NLP and supervised machine learning.
---

## 📖 Overview
This project aims to **detect cyberbullying** and offensive language on social media platforms using Natural Language Processing (NLP) and machine learning models.  
It demonstrates an end-to-end ML pipeline — from text preprocessing and feature extraction to model training and web deployment using Flask.

---

## 🚀 Features
- Detects and classifies online text as safe or harmful.  
- Preprocessing includes tokenization, stemming, and TF-IDF vectorization.  
- Multiple ML models trained and compared:
  - Logistic Regression  
  - XGBoost  
  - Naive Bayes  
  - Random Forest  
  - Decision Tree  
- Achieved **92% accuracy (F1 = 0.89)** on 72K+ labeled posts.  
- Real-time predictions through a simple Flask web app.  
- Generates a **bullying severity score** for interpretability.

---

## 🧰 Tech Stack
**Languages:** Python  
**Libraries:** Scikit-learn, XGBoost, Pandas, NLTK, Matplotlib, Seaborn  
**Framework:** Flask  
**Environment:** Jupyter Notebook / VS Code  

---

## 📈 Model Performance
| Model | Accuracy | F1-Score |
|--------|-----------|----------|
| Logistic Regression | 91.8% | 0.89 |
| XGBoost | 91.4% | 0.88 |
| Naive Bayes | 90.3% | 0.87 |
| Random Forest | 89.9% | 0.86 |
| Decision Tree | 88.2% | 0.84 |

✅ **Best Model:** Logistic Regression + TF-IDF  
✅ **Final Ensemble Accuracy:** ~92%

---

## 💻 Run Locally
Clone the project:
```bash
git clone https://github.com/vuchau/threat-detection-ml.git
cd threat-detection-ml
