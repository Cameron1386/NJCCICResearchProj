# will handle all text preproecessing tasks
# preprocessing.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import nltk

# Make sure to download stopwords if you haven't
nltk.download('stopwords')

def load_and_clean_data(file_path):
    # Load dataset
    data = pd.read_csv(file_path)

    # Basic text cleaning and stop word removal
    stop_words = set(stopwords.words('english'))
    data['cleaned_text'] = data['text'].apply(lambda x: ' '.join(
        word for word in x.lower().split() if word not in stop_words))
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        data['cleaned_text'], data['label'], test_size=0.2, random_state=42)
    
    # Convert text to TF-IDF features
    tfidf = TfidfVectorizer(max_features=1000)
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)
    
    return X_train_tfidf, X_test_tfidf, y_train, y_test
