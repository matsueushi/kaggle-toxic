# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from gensim import corpora, models
from gensim.matutils import corpus2dense
from keras.preprocessing.text import text_to_word_sequence
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score


# %%
train = pd.read_csv('../input/train_prepro.csv')
test = pd.read_csv('../input/test_prepro.csv')
train_comments = train['preprocessed_comment'].astype(str)
test_comments = test['preprocessed_comment'].astype(str)


# %%
dictionary = corpora.Dictionary.load('comment_dictionary.dict')
print(dictionary)
dictionary.filter_extremes(no_below=3, no_above=0.8, keep_n=300000)
print(dictionary)


# %%
all_corpus = train_comments.append(test_comments)
print(all_corpus.head())
all_bows = [dictionary.doc2bow(text_to_word_sequence(text))
            for text in all_corpus]
print(all_bows[3])

# %%
tfidf_model = models.TfidfModel(all_bows)
tfidf_corpus = tfidf_model[all_bows]
print(tfidf_corpus)


# %%
t_matrix = corpus2dense(tfidf_corpus, num_terms=len(dictionary.token2id)).T
print(t_matrix[0])
classifier = LogisticRegression()


# %%
