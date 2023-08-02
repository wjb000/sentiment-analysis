import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.classify.util import accuracy

# Download the movie reviews corpus
nltk.download('movie_reviews')

# Load the movie reviews corpus
documents = [(list(movie_reviews.words(fileid)), category)
              for category in movie_reviews.categories()
              for fileid in movie_reviews.fileids(category)]

# Shuffle the documents
import random
random.shuffle(documents)

# Define a feature extractor
# Here we're using the 2000 most common words as features
all_words = nltk.FreqDist(w.lower() for w in movie_reviews.words())
word_features = list(all_words)[:2000]

def document_features(document):
    document_words = set(document)
    features = {}
    for word in word_features:
        features['contains({})'.format(word)] = (word in document_words)
    return features

# Get the features for each document
featuresets = [(document_features(d), c) for (d,c) in documents]

# Split the data into training and testing sets
train_set, test_set = featuresets[100:], featuresets[:100]

# Train a NaiveBayes classifier
classifier = NaiveBayesClassifier.train(train_set)

# Test the classifier
print("Accuracy: ", accuracy(classifier, test_set))

# Show the most important features
classifier.show_most_informative_features(5)
