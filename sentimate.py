import requests
import nltk
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

# Constants
NEWSAPI_KEY = "YOUR_NEWSAPI_KEY"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_articles(ticker):
    query = ticker + " stock"
    params = {
        'q': query,
        'apiKey': NEWSAPI_KEY,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100
    }
    response = requests.get(NEWSAPI_ENDPOINT, params=params)
    if response.status_code != 200:
        print("Error fetching news:", response.text)
        return []

    articles = response.json().get('articles', [])
    return [article['description'] or article['title'] for article in articles if article['description'] or article['title']]

# Fetch articles for a ticker
ticker = input("Enter the stock ticker for which you want to fetch news articles: ")
articles = fetch_articles(ticker)

# For demonstration, manually label the first N articles
N = 10  # or however many you're comfortable labeling
print("\nManually label the first few articles:")
documents = []
for i, article in enumerate(articles[:N]):
    print(f"\nArticle {i+1}:\n{article}")
    label = input("Enter sentiment (positive/negative/neutral): ")
    documents.append((word_tokenize(article), label))

# Define a feature extractor
all_words = nltk.FreqDist(w.lower() for words, sentiment in documents for w in words)
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Train a classifier using the manually labeled articles
featuresets = [(document_features(d), c) for (d, c) in documents]
classifier = NaiveBayesClassifier.train(featuresets)

# Analyze the remaining articles
for article in articles[N:]:
    words = word_tokenize(article)
    sentiment = classifier.classify(document_features(words))
    print(f"\nArticle: {article}\nPredicted Sentiment: {sentiment}\n{'-' * 50}")
