import requests
from datetime import datetime, timedelta
from newspaper import Article
from transformers import pipeline

NEWSAPI_KEY = "6552cb40d51d4d22ad84c57a3d8d5a88"
NEWSAPI_ENDPOINT = "https://newsapi.org/v2/everything"

def fetch_articles(ticker):
    current_time = datetime.utcnow()
    past_week = (current_time - timedelta(days=7)).strftime("%Y-%m-%dT%H:%M:%SZ")
    
    query = ticker + " stock"
    params = {
        'q': query,
        'apiKey': NEWSAPI_KEY,
        'language': 'en',
        'sortBy': 'relevancy',
        'pageSize': 100,
        'from': past_week,
        'to': current_time.strftime("%Y-%m-%dT%H:%M:%SZ")
    }
    articles = []
    
    while True:
        response = requests.get(NEWSAPI_ENDPOINT, params=params)
        
        if response.status_code != 200:
            print("Error fetching news:", response.text)
            break
        
        data = response.json()
        articles.extend([(article['title'], article['url']) for article in data['articles'] if article['title'] and article['url']])
        
        if 'nextPage' not in data:
            break
        
        params['page'] = data['nextPage']
    
    return articles

def extract_text(url):
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text
    except:
        return ""

def is_related_to_company(text, company):
    return company.lower() in text.lower()

def sentiment_analysis(text):
    sentiment_analyzer = pipeline("sentiment-analysis", model="mrm8488/distilroberta-finetuned-financial-news-sentiment-analysis", truncation=True, max_length=512)
    sentiment = sentiment_analyzer(text[:512])[0]
    return sentiment['label']

ticker = input("Enter the stock ticker for which you want to fetch news articles: ")
articles = fetch_articles(ticker)

if not articles:
    print("No articles found for the given ticker in the past week.")
    exit()

sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}

print(f"\nAnalyzing sentiment for relevant articles:")
relevant_articles = 0
for i, (title, url) in enumerate(articles):
    text = extract_text(url)
    if text and is_related_to_company(text, ticker):
        sentiment = sentiment_analysis(text)
        sentiment_counts[sentiment] += 1
        relevant_articles += 1
        print(f"Article {relevant_articles}: {title}")
        print(f"URL: {url}")
        print(f"Sentiment: {sentiment}")
        print("-" * 50)
    elif is_related_to_company(title, ticker):
        sentiment = sentiment_analysis(title)
        sentiment_counts[sentiment] += 1
        relevant_articles += 1
        print(f"Article {relevant_articles}: {title}")
        print(f"URL: {url}")
        print("Failed to extract article content. Sentiment based on the title.")
        print(f"Sentiment: {sentiment}")
        print("-" * 50)

if relevant_articles == 0:
    print("No relevant articles found for the given ticker in the past week.")
else:
    print(f"\nSentiment Distribution for {ticker} based on {relevant_articles} relevant articles from the past week:")
    for sentiment, count in sentiment_counts.items():
        percentage = (count / relevant_articles) * 100
        print(f"{sentiment.capitalize()}: {count} articles ({percentage:.2f}%)")
