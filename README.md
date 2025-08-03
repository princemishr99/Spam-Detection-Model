# Spam-Detection-Model

## 📌 Project Overview
This machine learning project classifies SMS messages as **spam** or **ham (not spam)** using natural language processing techniques and a text classification model.

## 🧠 Techniques Used
- Text preprocessing (removing punctuation, lowercase)
- TF-IDF Vectorization
- Naive Bayes Classifier (MultinomialNB)
- Evaluation using accuracy, precision, recall

## 📂 Dataset
- **Source**: [Kaggle - SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
- File used: `spam.csv`

## 📈 Project Steps
1. Load and clean the dataset
2. Preprocess the text data (lowercase, remove punctuation)
3. Convert text into numeric format using TF-IDF
4. Train a Naive Bayes model
5. Test the model with sample messages
6. Optional: Deployed as Streamlit app on localhost

## 🧪 Output Example
User enters an SMS text → model returns **Spam** or **Not Spam**.
