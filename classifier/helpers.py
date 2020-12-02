# %%
import pandas as pd
import numpy as np
import pickle
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import pos_tag
from nltk.stem.porter import PorterStemmer
import re

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VS
from textstat import textstat

stemmer = PorterStemmer()
sentiment_analyzer = VS()

REGEX_SPACE_PATTERN = '\s+'
REGEX_URL_PATTERN = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
    '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
REGEX_MENTION_PATTERN = '@[\w\-]+'
REGEX_HASHTAG_PATTERN = '#[\w\-]+'

def preprocess(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned
    """
    parsed_text = re.sub(REGEX_SPACE_PATTERN, ' ', text_string)
    parsed_text = re.sub(REGEX_URL_PATTERN, 'URLHERE', parsed_text)
    parsed_text = re.sub(REGEX_MENTION_PATTERN, 'MENTIONHERE', parsed_text)
    return parsed_text

def tokenize(tweet):
    """Removes punctuation & excess whitespace, sets to lowercase,
    and stems tweets. Returns a list of stemmed tokens."""
    tweet = " ".join(re.split("[^a-zA-Z]*", tweet.lower())).strip()
    tokens = [stemmer.stem(t) for t in tweet.split()]
    return tokens

def basic_tokenize(tweet):
    """Same as tokenize but without the stemming"""
    tweet = " ".join(re.split("[^a-zA-Z.,!?]*", tweet.lower())).strip()
    return tweet.split()


def count_twitter_objs(text_string):
    """
    Accepts a text string and replaces:
    1) urls with URLHERE
    2) lots of whitespace with one instance
    3) mentions with MENTIONHERE
    4) hashtags with HASHTAGHERE

    This allows us to get standardized counts of urls and mentions
    Without caring about specific people mentioned.
    
    Returns counts of urls, mentions, and hashtags.
    """
    parsed_text = re.sub(REGEX_SPACE_PATTERN, ' ', text_string)
    parsed_text = re.sub(REGEX_URL_PATTERN, 'URLHERE', parsed_text)
    parsed_text = re.sub(REGEX_MENTION_PATTERN, 'MENTIONHERE', parsed_text)
    parsed_text = re.sub(REGEX_HASHTAG_PATTERN, 'HASHTAGHERE', parsed_text)
    return {
        'urls': parsed_text.count('URLHERE'),
        'mentions': parsed_text.count('MENTIONHERE'),
        'hashtags': parsed_text.count('HASHTAGHERE')
        }

def miscellaneous_features(tweet, features):
    """This function takes a string and returns a list of features.
    These include Sentiment scores, Text and Readability scores,
    as well as Twitter specific features"""
    sentiment = sentiment_analyzer.polarity_scores(tweet)
    
    words = preprocess(tweet)
    
    syllables = textstat.syllable_count(words)
    num_chars = sum(len(w) for w in words)
    num_chars_total = len(tweet)
    num_terms = len(tweet.split())
    num_words = len(words.split())
    avg_syl = round(float((syllables+0.001))/float(num_words+0.001),4)
    num_unique_terms = len(set(words.split()))
    
    # Modified FK grade, where avg words per sentence is just num words/1
    FKRA = round(float(0.39 * float(num_words)/1.0) + float(11.8 * avg_syl) - 15.59,1)
    # Modified FRE score, where sentence fixed to 1
    FRE = round(206.835 - 1.015*(float(num_words)/1.0) - (84.6*float(avg_syl)),2)
    
    twitter_objs = count_twitter_objs(tweet)
    retweet = 0
    if "rt" in words:
        retweet = 1

    feature_dict = {
        'FKRA': FKRA,
        'FRE': FRE,
        'num_syllables': syllables,
        'avg_syl_per_word': avg_syl,
        'num_chars': num_chars,
        'num_chars_total': num_chars_total,
        'num_terms': num_terms,
        'num_words': num_words,
        'num_unique_terms': num_unique_terms,
        'sentiment_neg': sentiment['neg'],
        'sentiment_pos': sentiment['pos'],
        'sentiment_neu': sentiment['neu'],
        'sentiment_compound': sentiment['compound'],
        'num_hashtags': twitter_objs['hashtags'], 
        'num_mentions': twitter_objs['mentions'], 
        'num_urls': twitter_objs['urls'], 
        'is_retweet': retweet,
    }

    return [feature_dict[x] for x in features]

def get_miscellaneous_features(tweets, features):
    output = []
    for t in tweets:
        output.append(miscellaneous_features(t, features))
    return np.array(output)


def get_pos_tags(tweets):
    """Takes a list of strings (tweets) and
    returns a list of strings of (POS tags).
    """
    tweet_tags = []
    for t in tweets:
        tokens = basic_tokenize(preprocess(t))
        tags = pos_tag(tokens)
        tag_list = [x[1] for x in tags]
        tag_str = " ".join(tag_list)
        tweet_tags.append(tag_str)
    return tweet_tags


def transform_inputs(tweets, ngram_model, idf_vector, pos_model):
    """
    This function takes a list of tweets, along with used to
    transform the tweets into the format accepted by the model.

    Each tweet is decomposed into
    (a) An array of TF-IDF scores for a set of character n-grams in the tweet.
    (b) An array of TF-IDF scores for a set of POS tag sequences in the tweet.
    (c) An array of features including sentiment, vocab, and readability.

    Returns a pandas dataframe where each row is the set of features
    for a tweet. The features are a subset selected using a Logistic
    Regression with L1-regularization on the training data.

    """

    with open("../pickled_models/final_misc_features.pkl", "rb") as file:
        misc_feature_names = pickle.load(file)

    ngram_tf_array = ngram_model.fit_transform(tweets).toarray()
    ngram_tfidf_array = ngram_tf_array * idf_vector
    print ("Built N-gram features")

    pos_tags = get_pos_tags(tweets)
    pos_array = pos_model.fit_transform(pos_tags).toarray()
    print ("Built POS features")

    misc_array = get_miscellaneous_features(tweets, misc_feature_names)
    print ("Built miscellaneous features")

    feature_matrix = np.concatenate([ngram_tfidf_array, pos_array, misc_array], axis=1)
    return pd.DataFrame(feature_matrix)

def get_tweets_predictions(tweets, perform_prints=True):
    
    print ("Loading models...")
    model = pickle.load(open('../pickled_models/final_model.pkl', "rb"))
    ngram_model = pickle.load(open("../pickled_models/ngram_model.pkl", "rb" ))
    idf_vector = pickle.load(open('../pickled_models/idf_model.pkl', 'rb'))
    pos_model = pickle.load(open('../pickled_models/pos_model.pkl', 'rb'))

    X = transform_inputs(tweets, ngram_model, idf_vector, pos_model)

    print ("Running classification model...")
    predictions = model.predict(X)

    return predictions


def class_to_name(class_label):
    """
    This function can be used to map a numeric
    feature name to a particular class.
    """
    class_dict = {
        0: "Hate Speech",
        1: "Offensive Language",
        2: "Neither",
    }

    if class_label in class_dict.keys():
        return class_dict[class_label]
    else:
        return "No Label"