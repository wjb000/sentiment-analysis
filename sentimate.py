import requests
import nltk
from textblob import TextBlob
from nltk.tokenize import word_tokenize
from nltk.classify import NaiveBayesClassifier

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

def get_sentiment(text):
    analysis = TextBlob(text)
    # Define sentiment based on polarity
    if analysis.sentiment.polarity > 0.1:
        return 'positive'
    elif analysis.sentiment.polarity < -0.1:
        return 'negative'
    else:
        return 'neutral'

ticker = input("Enter the stock ticker for which you want to fetch news articles: ")
articles = fetch_articles(ticker)

N = 10  # or however many you're comfortable labeling
print("\nAuto-labeling the first few articles using TextBlob:")
documents = []
for i, article in enumerate(articles[:N]):
    label = get_sentiment(article)
    documents.append((word_tokenize(article), label))

all_words = nltk.FreqDist(w.lower() for words, sentiment in documents for w in words)
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

featuresets = [(document_features(d), c) for (d, c) in documents]
classifier = NaiveBayesClassifier.train(featuresets)

for article in articles[N:]:
    words = word_tokenize(article)
    sentiment = classifier.classify(document_features(words))
    print(f"\nArticle: {article}\nPredicted Sentiment: {sentiment}\n{'-' * 50}")
