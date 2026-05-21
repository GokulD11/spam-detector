# 📩 SMS Spam Detector

An end-to-end machine learning project that classifies SMS messages as **spam** or **ham** (legitimate), served as a real-time web app.

🔗 **[Live Demo](https://spam-detector-gjjfnytjnrp6v7dmbjb5xq.streamlit.app/)** &nbsp;|&nbsp; Built with scikit-learn + Streamlit

![App Screenshot](screenshots/app_demo.png)


---

## 📋 Overview

This project covers the full ML pipeline: data cleaning, exploratory analysis, NLP preprocessing, feature engineering, model comparison, evaluation, and deployment as an interactive web application. A user can type any message and instantly get a spam/ham prediction.

## 🎯 Problem

Spam detection is a **supervised binary text classification** problem. Given the text of an SMS, predict whether it is spam (unwanted/promotional) or ham (legitimate). The dataset is **imbalanced** (~87% ham, ~13% spam), which shaped the choice of evaluation metrics.

## 📊 Dataset

- **Source:** [UCI SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- **Size:** 5,572 messages (5,169 after removing duplicates)
- **Classes:** ham (legitimate) and spam

## 🔬 Approach

| Stage | What I did |
|-------|------------|
| **EDA** | Found class imbalance, spam messages ~2× longer than ham, distinct spam vocabulary (FREE, WIN, CLAIM) |
| **Preprocessing** | Lowercasing, regex punctuation removal, NLTK tokenization, stopword removal, Porter stemming |
| **Features** | TF-IDF vectorization (top 3,000 terms) |
| **Models** | Compared Multinomial Naive Bayes, Logistic Regression, Linear SVM |
| **Validation** | Stratified train/test split + 5-fold cross-validation |
| **Deployment** | Saved pipeline with joblib, served via Streamlit Cloud |

## 📈 Results

| Model | Accuracy | Precision | Recall | F1 |
|-------|----------|-----------|--------|-----|
| Naive Bayes | 0.9718 | 1.0000 | 0.7786 | 0.8755 |
| Logistic Regression | 0.9583 | 0.9681 | 0.6947 | 0.8089 |
| **Linear SVM** ✅ | **0.9816** | **0.9746** | **0.8779** | **0.9237** |

**Chosen model: Linear SVM** — best F1 and recall while keeping precision high.

> **Key decision:** For spam, I prioritized **precision** because a false positive (sending a legitimate message to junk) is far costlier than a false negative (one spam reaching the inbox). Linear SVM gave the best balance with precision above 97%.

## 🗂️ Project Structure

```
spam-detector/
├── app/
│   ├── app.py              # Streamlit web app
│   └── preprocessing.py    # Shared text preprocessing
├── data/                   # Dataset (raw + cleaned)
├── models/
│   └── spam_classifier.joblib   # Trained pipeline (TF-IDF + Linear SVM)
├── notebooks/
│   ├── 01_eda.ipynb        # Exploratory data analysis
│   ├── 02_preprocessing.ipynb   # Text cleaning pipeline
│   └── 03_modeling.ipynb   # Feature engineering, training, evaluation
├── requirements.txt
└── README.md
```

## 🚀 Run Locally

```bash
# Clone the repo
git clone https://github.com/GokulD11/spam-detector.git
cd spam-detector

# Create and activate a virtual environment
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app/app.py
```

The app will open at `http://localhost:8501`.

## 🛠️ Tech Stack

Python · pandas · scikit-learn · NLTK · Streamlit · matplotlib · seaborn

## 🧠 Key Concepts Demonstrated

- **NLP preprocessing pipeline** — tokenization, stopword removal, stemming
- **TF-IDF feature engineering** — weighting terms by discriminative power
- **Data leakage prevention** — fit the vectorizer on training data only, transform both
- **Metric selection for imbalanced data** — precision/recall/F1 over raw accuracy
- **Model serialization & deployment** — bundled pipeline avoids train/serve skew

## 🔮 Future Improvements

- Add confidence scores via `CalibratedClassifierCV`
- Experiment with n-grams and lemmatization
- Try transformer-based models (e.g. DistilBERT) for comparison
- Add unit tests for the preprocessing pipeline

## 👤 Author

**Gokul D** · [GitHub](https://github.com/GokulD11) · [LinkedIn](https://www.linkedin.com/in/gokul-d15/)