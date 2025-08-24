import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

df_original = pd.read_csv(
    'https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv',
    sep='\t', header=None, names=['label', 'message']
)
df_original['label'] = df_original['label'].map({'ham': 0, 'spam': 1})

df_kaggle = pd.read_csv('spam.csv', encoding='latin-1')[['v1', 'v2']]
df_kaggle.columns = ['label', 'message']
df_kaggle['label'] = df_kaggle['label'].str.strip().map({'ham': 0, 'spam': 1})
df_kaggle.dropna(subset=['label'], inplace=True)
df_kaggle['label'] = df_kaggle['label'].astype(int)

df = pd.concat([df_original, df_kaggle], ignore_index=True)
df['message'] = df['message'].apply(clean_text)


X_train, X_test, y_train, y_test = train_test_split(
    df['message'], df['label'], test_size=0.2, random_state=42
)

vectorizer = TfidfVectorizer(
    stop_words='english',
    ngram_range=(1, 2),
    max_df=0.95,
    min_df=2,
    max_features=10000
)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ================ Logistic Regression ================
log_model = LogisticRegression(max_iter=1000)
log_model.fit(X_train_tfidf, y_train)
log_pred = log_model.predict(X_test_tfidf)
log_acc = accuracy_score(y_test, log_pred)
log_cm = confusion_matrix(y_test, log_pred)
with open('spam_model_logistic.pkl', 'wb') as f:
    pickle.dump((log_model, vectorizer), f)

# ================ Naive Bayes ================
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)
nb_pred = nb_model.predict(X_test_tfidf)
nb_acc = accuracy_score(y_test, nb_pred)
nb_cm = confusion_matrix(y_test, nb_pred)
with open('spam_model_naivebayes.pkl', 'wb') as f:
    pickle.dump((nb_model, vectorizer), f)

# ================ Random Forest ================
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_tfidf, y_train)
rf_pred = rf_model.predict(X_test_tfidf)
rf_acc = accuracy_score(y_test, rf_pred)
rf_cm = confusion_matrix(y_test, rf_pred)
with open('spam_model_randomforest.pkl', 'wb') as f:
    pickle.dump((rf_model, vectorizer), f)

# ================ Print results ================
print("\n📊 [Logistic Regression]")
print("✅ Accuracy:", round(log_acc * 100, 2), "%")
print("Confusion Matrix:\n", log_cm)

print("\n📊 [Naive Bayes]")
print("✅ Accuracy:", round(nb_acc * 100, 2), "%")
print("Confusion Matrix:\n", nb_cm)

print("\n📊 [Random Forest]")
print("✅ Accuracy:", round(rf_acc * 100, 2), "%")
print("Confusion Matrix:\n", rf_cm)
