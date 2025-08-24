# Email Spam Classifier

This is a simple web application built with Flask to classify emails as Spam or Not Spam.

## Features
- Classify email text.
- Choose between different machine learning models (Logistic Regression, Naive Bayes, Random Forest).
- Shows prediction confidence, word count, and the most influential word.

## How to Run
1. Install the required libraries: `pip install -r requirements.txt`
2. Run the training script to generate the models: `python train_model.py`
3. Run the Flask app: `python app.py`
