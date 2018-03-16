# -*- coding: utf-8 -*-
# %%
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


# %%
train = pd.read_csv('../input/train_prepro.csv')
test = pd.read_csv('../input/test_prepro.csv')


# %%
print(train)


# %%
train_preprocessed_comment = train['preprocessed_comment_text'].astype(str)
test_preprocessed_comment = test['preprocessed_comment_text'].astype(str)


# %%
# Logistic Regressionここから
# TfidfVectorizer
tfidf_vec = TfidfVectorizer(max_features=50000, min_df=2, max_df=0.5)
train_dtm = tfidf_vec.fit_transform(train_preprocessed_comment)
test_dtm = tfidf_vec.transform(test_preprocessed_comment)


# %%
print(train_dtm.shape)
print(test_dtm.shape)


# %%
