# Importing
import nltk
import random
import re
import string
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tag import pos_tag
from nltk.tokenize import word_tokenize
from nltk import FreqDist, classify, NaiveBayesClassifier

#NLTK data downloaded
nltk.download(['movie_reviews', 'stopwords', 'wordnet', 'punkt', 'averaged_perceptron_tagger_eng'], quiet=True)




# Function to remove noise (links, mentions, punctuation) and lemmatize
def process_text(text_tokens, stop_words):
    cleaned_tokens = []
    lemmatizer = WordNetLemmatizer()
    for i, (word, tag) in enumerate(pos_tag(text_tokens)):
        # Remove hyperlinks
        word = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+#]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', word)
        # Remove mentions
        word = re.sub(r"(@[A-Za-z0-9_]+)", "", word)

        if not word:  # Skip if word became empty after cleaning
            continue

        # Determine POS for lemmatization
        if tag.startswith('NN'):
            pos = 'n'
        elif tag.startswith('VB'):
            pos = 'v'
        else:
            pos = 'a'

        # Negation handling: Prepend 'NOT_' if preceded by a negation word
        if i > 0 and text_tokens[i - 1].lower() in ["not", "no", "n't", "never"]:
            word = "NOT_" + word

        lemmatized_word = lemmatizer.lemmatize(word, pos)

        if len(lemmatized_word) > 0 and lemmatized_word not in stopwords.words(
                'english') and lemmatized_word not in set(string.punctuation):
            cleaned_tokens.append(lemmatized_word.lower())
    return cleaned_tokens


# Load movie reviews data
positive_reviews = movie_reviews.fileids('pos')
negative_reviews = movie_reviews.fileids('neg')

# Create a list of all words from all reviews for feature extraction
all_words = []
for fileid in positive_reviews:
    for word in movie_reviews.words(fileid):
        all_words.append(word.lower())
for fileid in negative_reviews:
    for word in movie_reviews.words(fileid):
        all_words.append(word.lower())

# Get stop words
stop_words = set(stopwords.words('english'))

# Preprocess and prepare data for the model
documents = []
for fileid in positive_reviews:
    tokens = process_text(movie_reviews.words(fileid), stop_words)
    documents.append((tokens, 'pos'))

for fileid in negative_reviews:
    tokens = process_text(movie_reviews.words(fileid), stop_words)
    documents.append((tokens, 'neg'))

# Shuffle the documents
random.shuffle(documents)




# Function to extract features (Bag-of-Words)
def document_features(document, word_features):
    document_words = set(document)
    features = {}
    for word in word_features:
        features[word] = (word in document_words)
    return features


# Get a frequency distribution of all words to select top features
all_words_freq = FreqDist(all_words)
word_features = list(all_words_freq.keys())[:2000]  # Use top 2000 words as features

# Create feature sets for training and testing
featuresets = [(document_features(d, word_features), category) for (d, category) in documents]

# Split data into training and testing sets
train_set, test_set = featuresets[1000:], featuresets[:1000]


# Train a Naive Bayes Classifier
classifier = NaiveBayesClassifier.train(train_set)

# Evaluate the classifier
accuracy = classify.accuracy(classifier, test_set)
print(f"Classifier Accuracy: {accuracy:.2f}")

# Show most informative features
print("Most informative features:")
classifier.show_most_informative_features(15)




def analyze_sentiment(text_to_analyze, classifier, word_features, stop_words):
    custom_tokens = process_text(word_tokenize(text_to_analyze), stop_words)
    custom_features = document_features(custom_tokens, word_features)
    return classifier.classify(custom_features)


# Test with a custom sentence
custom_sentence_positive = "This movie was absolutely fantastic, a true masterpiece!"
sentiment_positive = analyze_sentiment(custom_sentence_positive, classifier, word_features, stop_words)
print(f"\nSentiment for '{custom_sentence_positive}': {sentiment_positive}")

custom_sentence_negative = "I hated every single moment, it was not good at all."
sentiment_negative = analyze_sentiment(custom_sentence_negative, classifier, word_features, stop_words)
print(f"Sentiment for '{custom_sentence_negative}': {sentiment_negative}")

custom_sentence_sarcasm = "What a brilliant idea, flying me to the wrong city! Excellent service."
sentiment_sarcasm = analyze_sentiment(custom_sentence_sarcasm, classifier, word_features, stop_words)
print(f"Sentiment for '{custom_sentence_sarcasm}': {sentiment_sarcasm}")
