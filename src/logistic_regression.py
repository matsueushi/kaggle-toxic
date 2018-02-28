# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from gensim import corpora, models
from keras.preprocessing.text import text_to_word_sequence


# %%
train = pd.read_csv('../input/train_prepro.csv')
test = pd.read_csv('../input/test_prepro.csv')
train_comments = train['preprocessed_comment'].astype(str)
test_comments = test['preprocessed_comment'].astype(str)


# %%
dictionary = corpora.Dictionary.load('comment_dictionary.dict')
print(dictionary)
corpus = [dictionary.doc2bow(text_to_word_sequence(t)) for t in train_comments]

# %%
tfidf_model = models.TfidfModel(corpus)
corpus_tfidf = tfidf_model[corpus]


# %%
print(corpus_tfidf[10])
