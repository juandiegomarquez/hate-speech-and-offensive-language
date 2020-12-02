# %% [markdown]
# # Hate Speech Model

import pandas as pd
import numpy as np
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk import corpus

from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import classification_report
from sklearn.svm import LinearSVC

import helpers

df = pd.read_csv('../data/labeled_data.csv')
tweets = df.tweet

# %% [markdown]
# ## Part 1: Compute full feature matrix
# Features: <br>
#  - tfidf character n-gram values:
#      - character unigrams
#      - character bigrams
#      - character trigrams
#  - tfidf part of speech values:
#      - pos unigrams
#      - pos bigrams
#      - pos trigrams
#  - miscellaneous features:
#      - FKRA (Flesch-Kincaid Reading Age)
#           - (0.39 x num_words) + (11.8 x avg_syl) - 15.59
#      - FRE (Flesch Reading Ease)
#           - 206.835 - (1.015 x num_words) - (84.6 x avg_syl)
#      - total number of syllables
#      - avg number of syllables
#      - number of characters in words (excludes spaces, punctuation, etc)
#      - total number of characters in document
#      - number of terms (includes non-word terms such as punctuation, emojis, etc)
#      - number of words
#      - number of unique terms
#      - vader negative sentiment
#      - vader positive sentiment
#      - vader neutral sentiment
#      - vader compound sentiment
#      - number of hashtags
#      - number of urls
#      - number of mentions
#      - boolean retweet/non-retweet

stopwords = corpus.stopwords.words("english")
additional_exclusions = ["#ff", "ff", "rt"]
stopwords.extend(additional_exclusions)

ngram_vectorizer = TfidfVectorizer(
    tokenizer=helpers.tokenize,
    preprocessor=helpers.preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords,
    use_idf=True,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=10000,
    min_df=5,
    max_df=0.501
    )

# Tfidf feature matrix
# Also save idf feature matrix
ngram_features = ngram_vectorizer.fit_transform(tweets).toarray()
ngram_vocab = {v:i for i, v in enumerate(ngram_vectorizer.get_feature_names())}
ngram_idf_values = ngram_vectorizer.idf_

# Compute parts of speech (pos) for all tweets
tweet_tags = helpers.get_pos_tags(tweets)

# Use the TFIDF vectorizer to get a token matrix for the POS tags
pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    max_features=5000,
    min_df=5,
    max_df=0.501,
    )

# POS feature matrix
pos_features = pos_vectorizer.fit_transform(pd.Series(tweet_tags)).toarray()
pos_vocab = {v:i for i, v in enumerate(pos_vectorizer.get_feature_names())}

misc_feature_names = ["FKRA",
                        "FRE",
                        "num_syllables",
                        "avg_syl_per_word",
                        "num_chars",
                        "num_chars_total",
                        "num_terms",
                        "num_words",
                        "num_unique_terms",
                        "sentiment_neg",
                        "sentiment_pos",
                        "sentiment_neu", 
                        "sentiment_compound", 
                        "num_hashtags",
                        "num_mentions",
                        "num_urls",
                        "is_retweet"
                        ]

misc_features = helpers.get_miscellaneous_features(tweets, features = misc_feature_names)

# Complete feature matrix
feature_matrix = np.concatenate([ngram_features,
                                pos_features,
                                misc_features
                                ] ,axis=1)


ngram_feature_names = ['']*len(ngram_vocab)
for k,v in ngram_vocab.items():
    ngram_feature_names[v] = k


pos_feature_names = ['']*len(pos_vocab)
for k,v in pos_vocab.items():
    pos_feature_names[v] = k

feature_names = ngram_feature_names + pos_feature_names + misc_feature_names

X = pd.DataFrame(feature_matrix)
y = df['class'].astype(int)

# %% [markdown]
# ## Part 2: Feature Selection and Train Model

feature_selector = SelectFromModel(
                            LogisticRegression(
                            class_weight='balanced',
                            penalty='l2',
                            C=0.01
                            )
                    )
X_selected = feature_selector.fit_transform(X,y)

model = LinearSVC(class_weight='balanced',
                  C=0.01,
                  penalty='l2',
                  loss='squared_hinge',
                  multi_class='ovr'
                  ).fit(X_selected, y)
                  
y_preds = model.predict(X_selected)

report = classification_report(y, y_preds)
print(report)

# Feature indices
final_feature_indices = feature_selector.get_support(indices=True) 
# Feature names
final_feature_list = [str(feature_names[i]) for i in final_feature_indices]

# %% [markdown]
# # Part 3: Package models
# Above we computed a full feature matrix, selected the best features, <br>
# trained a model, and generated a report to evaluate the model. <br>
# Below we will train a model with only the selected features from the <br>
# start and regenerate the report to verify the results are the same.

select_ngram_features = final_feature_list[:final_feature_list.index('z z z')+1]
select_pos_features = final_feature_list[final_feature_list.index('z z z')+1:final_feature_list.index('VBZ DT JJ')+1]
select_misc_features = final_feature_list[final_feature_list.index('VBZ DT JJ')+1:]

with open("../pickled_models/final_misc_features.pkl", "wb") as file:
    pickle.dump(select_misc_features, file)

select_ngram_vocab = {v:i for i, v in enumerate(select_ngram_features)}
select_ngram_indices = final_feature_indices[:len(select_ngram_features)]

select_ngram_vectorizer = TfidfVectorizer(
    tokenizer=helpers.tokenize,
    preprocessor=helpers.preprocess,
    ngram_range=(1, 3),
    stop_words=stopwords, 
    use_idf=False,
    smooth_idf=False,
    norm=None, 
    decode_error='replace',
    min_df=1,
    max_df=1.0,
    vocabulary=select_ngram_vocab
    )

with  open('../pickled_models/ngram_model.pkl',"wb") as file:
    pickle.dump(select_ngram_vectorizer, file)

select_ngram_features = select_ngram_vectorizer.fit_transform(tweets).toarray()
select_ngram_idf_values = ngram_idf_values[select_ngram_indices]

with open('../pickled_models/idf_model.pkl',"wb") as file:
    pickle.dump(select_ngram_idf_values, file)

final_ngram_features = select_ngram_features * select_ngram_idf_values

select_pos_vocab = {v:i for i, v in enumerate(select_pos_features)}
select_pos_vectorizer = TfidfVectorizer(
    tokenizer=None,
    lowercase=False,
    preprocessor=None,
    ngram_range=(1, 3),
    stop_words=None,
    use_idf=False,
    smooth_idf=False,
    norm=None,
    decode_error='replace',
    min_df=1,
    max_df=1.0,
    vocabulary=select_pos_vocab
    )

with open('../pickled_models/pos_model.pkl',"wb") as file:
    pickle.dump(select_pos_vectorizer, file)

final_pos_features = select_pos_vectorizer.fit_transform(tweet_tags).toarray()
final_misc_features = helpers.get_miscellaneous_features(tweets,
                                                         features=select_misc_features)

final_feature_matrix = np.concatenate([final_ngram_features,
                                        final_pos_features,
                                        final_misc_features
                                        ], axis=1)
X_final = pd.DataFrame(final_feature_matrix)
y_preds_final = model.predict(X_final)

report = classification_report(y, y_preds_final)
print(report)
# %% [markdown]
# # Part 4: Save Model
with open('../pickled_models/final_model.pkl',"wb") as file:
    pickle.dump(model, file)
# %%
