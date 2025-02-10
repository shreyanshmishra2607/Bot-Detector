import pandas as pd
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report
from nltk.sentiment import SentimentIntensityAnalyzer

# Download necessary NLTK data
nltk.download('vader_lexicon')
sia = SentimentIntensityAnalyzer()

# Load dataset
data = pd.read_csv('bot_detection_data.csv')  # Update with actual filename

# Preprocessing function
def preprocess_text(text):
    text = str(text).lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'[^a-zA-Z ]', '', text)  # Remove special characters
    return text

data['clean_tweet'] = data['Tweet'].apply(preprocess_text)

# Extract sentiment scores
data['sentiment'] = data['clean_tweet'].apply(lambda x: sia.polarity_scores(x)['compound'])

# TF-IDF Feature Extraction
tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1,2))  # Increase features and add bigrams
tfidf_features = tfidf.fit_transform(data['clean_tweet']).toarray()

# Prepare final dataset
X = np.hstack((tfidf_features, data[['Retweet Count', 'Mention Count', 'Follower Count', 'sentiment']].values))
y = data['Bot Label']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model with Gradient Boosting for better accuracy
model = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
