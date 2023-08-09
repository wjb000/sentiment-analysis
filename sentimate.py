import requests
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

# Constants
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

# --- Setup for sentiment analysis ---

# [For demonstration, I'll use the mock dataset. Ideally, use a proper dataset]
documents = [([...], "positive"), ([...], "negative")]  # Example data format

# Preprocessing setup
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# Define a feature extractor
all_words = nltk.FreqDist(w.lower() for words, sentiment in documents for w in words)
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Train a classifier
featuresets = [(document_features(d), c) for (d, c) in documents]
classifier = NaiveBayesClassifier.train(featuresets)

# --- News fetching and analysis ---

def fetch_articles(ticker):
    query = ticker + " stock"  # Modify this query if needed
    params = {
        'q': query,
        'apiKey': NEWSAPI_KEY,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100  # Fetch 100 articles
    }
    response = requests.get(NEWSAPI_ENDPOINT, params=params)
    if response.status_code != 200:
        print("Error fetching news:", response.text)
        return []

    articles = response.json().get('articles', [])
    return [article['description'] or article['title'] for article in articles if article['description'] or article['title']]

def analyze_articles(articles):
    results = []
    for article in articles:
        words = word_tokenize(article)
        words = [lemmatizer.lemmatize(word) for word in words if word not in stop_words]
        features = document_features(words)
        sentiment = classifier.classify(features)
        results.append((article, sentiment))
    return results

def main():
    ticker = input("Enter the stock ticker for sentiment analysis: ")
    articles = fetch_articles(ticker)
    sentiments = analyze_articles(articles)

    positive_count, negative_count, neutral_count = 0, 0, 0

    # Display and count results
    for article, sentiment in sentiments:
        if sentiment == "positive":
            positive_count += 1
        elif sentiment == "negative":
            negative_count += 1
        else:
            neutral_count += 1
        print(f"Article: {article}\nSentiment: {sentiment}\n{'-' * 50}")

    # Summary
    print(f"\nSummary for {ticker}:")
    print(f"Positive articles: {positive_count}")
    print(f"Negative articles: {negative_count}")
    print(f"Neutral articles: {neutral_count}")

if __name__ == '__main__':
    main()
