#import libraries
import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Download necessary NLTK data
nltk.download('stopwords')
nltk.download('wordnet')
#Load dataset
print("Loading dataset...")
data_path = r"C:\jey\vamsi_krishna\complaints.csv"  
df = pd.read_csv(data_path).sample(n=150000, random_state=42)
print(f" Dataset loaded successfully. Total rows: {len(df)}")

if "Consumer complaint narrative" not in df.columns or "Product" not in df.columns:
    raise ValueError("Dataset must contain 'Consumer complaint narrative' and 'Product' columns.")
df = df[['Consumer complaint narrative', 'Product']]
df = df.rename(columns={'Consumer complaint narrative': 'complaint', 'Product': 'category'})
#Filter for 4 main categories
allowed_categories = [
    "Credit reporting", 
    "Debt collection", 
    "Consumer Loan", 
    "Mortgage"
]
df = df[df['category'].str.contains('|'.join(allowed_categories), case=False, na=False)]
#Clean text
df = df.dropna(subset=['complaint'])
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", "", text)
    words = text.split()
    words = [lemmatizer.lemmatize(w) for w in words if w not in stop_words]
    return " ".join(words)

print("Cleaning text data")
df['clean_complaint'] = df['complaint'].apply(clean_text)
#Split data
X = df['clean_complaint']
y = df['category']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Data prepared. Training samples: {len(X_train)}, Testing samples: {len(X_test)}")
# TF-IDF Vectorization
vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


#Train models

models = {}

print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=300)
lr.fit(X_train_tfidf, y_train)
y_pred_lr = lr.predict(X_test_tfidf)
models["Logistic Regression"] = lr
print(f"Logistic Regression Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")

print("\nTraining Naive Bayes...")
nb = MultinomialNB()
nb.fit(X_train_tfidf, y_train)
y_pred_nb = nb.predict(X_test_tfidf)
models["Naive Bayes"] = nb
print(f"Naive Bayes Accuracy: {accuracy_score(y_test, y_pred_nb):.4f}")

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=150, random_state=42)
rf.fit(X_train_tfidf, y_train)
y_pred_rf = rf.predict(X_test_tfidf)
models["Random Forest"] = rf
print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")

print("\n All models trained successfully!\n")
# Prediction Section
best_model = models["Logistic Regression"]  

print("Manual Predictions — type complaint text('exit' to quit):")
while True:
    user_input = input("\nEnter complaint text: ")
    if user_input.lower() == 'exit':
        break
    clean_input = clean_text(user_input)
    input_vec = vectorizer.transform([clean_input])
    pred_category = best_model.predict(input_vec)[0]
    print(f"Predicted Category: {pred_category}")
# sample predictions 
print("\nRunning 20 sample predictions:\n")
sample_complaints = [
    "My credit report shows incorrect late payments even though I paid on time.",
    "The bank keeps calling me about a loan I never took.",
    "I was charged an extra fee on my mortgage statement this month.",
    "Debt collectors are harassing me even after I cleared my dues.",
    "My loan application was denied without any valid reason.",
    "There is a wrong entry in my credit history that I cannot remove.",
    "Mortgage interest rate increased without prior notice.",
    "I have been billed twice for my credit card payment.",
    "I am getting constant calls from collection agencies.",
    "My credit score dropped drastically after a false report.",
    "They are refusing to update my loan repayment details.",
    "Debt collection agency threatened me with legal action.",
    "Mortgage payment not reflecting in my online account.",
    "Consumer loan processing is taking too long.",
    "My credit report has mixed information from another person.",
    "I got a message that my loan was approved but later cancelled.",
    "Unable to contact the credit reporting agency for corrections.",
    "The mortgage company applied a penalty by mistake.",
    "Debt collector is using abusive language in calls.",
    "Bank rejected my consumer loan even with good credit history."
]
for complaint in sample_complaints:
    clean_sample = clean_text(complaint)
    sample_vec = vectorizer.transform([clean_sample])
    prediction = best_model.predict(sample_vec)[0]
    print(f"Complaint: {complaint}")
    print(f"Predicted Category: {prediction}\n")