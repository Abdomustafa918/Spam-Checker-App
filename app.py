from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load all models
with open('spam_model_logistic.pkl', 'rb') as f:
    log_model, vectorizer = pickle.load(f)

with open('spam_model_naivebayes.pkl', 'rb') as f:
    nb_model, _ = pickle.load(f)

with open('spam_model_randomforest.pkl', 'rb') as f:
    rf_model, _ = pickle.load(f)

models = {
    "Logistic Regression": log_model,
    "Naive Bayes": nb_model,
    "Random Forest": rf_model
}

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    confidence = None
    word_count = None
    influential_word = None
    selected_model_name = "Logistic Regression"

    if request.method == 'POST':
        text = request.form['message']
        selected_model_name = request.form.get('model', 'Logistic Regression')
        selected_model = models.get(selected_model_name, log_model)

        word_count = len(text.split())

        # Vectorize input
        text_tfidf = vectorizer.transform([text])

        # Predict
        pred = selected_model.predict(text_tfidf)[0]

        # Confidence (if supported)
        try:
            proba = selected_model.predict_proba(text_tfidf)[0]
            confidence = f"{max(proba) * 100:.2f}%"
        except:
            confidence = "N/A"

        prediction = "Spam" if pred == 1 else "Not Spam"

        # Get top influential word by TF-IDF
        tfidf_vector = text_tfidf.toarray()[0]
        if tfidf_vector.any():
            max_index = np.argmax(tfidf_vector)
            influential_word = vectorizer.get_feature_names_out()[max_index]
        else:
            influential_word = "N/A"

    return render_template("index.html",
                           prediction=prediction,
                           confidence=confidence,
                           word_count=word_count,
                           influential_word=influential_word,
                           models=models.keys(),
                           selected_model=selected_model_name)

if __name__ == '__main__':
    app.run(debug=True)
