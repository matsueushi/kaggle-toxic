# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


# %%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# %%
def comment_preprocessing(comment, word_counts=None, count_threshold=10):
    def is_necessary(s, word_counts):
        ret = (not s.isdigit()) and len(s) > 1
        if word_counts == None:
            return ret
        else:
            if s in word_counts:
                return ret and (word_counts[s] > count_threshold)
            else:
                return False
    comment = comment.replace('\'', '')
    seq = text_to_word_sequence(remove_stopwords(comment).lower())
    seq = [s for s in seq if is_necessary(s, word_counts)]
    comment_str = ' '.join(seq)
    return comment_str


# %%
train_comments = train['comment_text']
tokenizer = Tokenizer()
fit_train_comments = [comment_preprocessing(comment)
                      for comment in train_comments]
tokenizer.fit_on_texts(fit_train_comments)
prepro_train_comments = [comment_preprocessing(comment, tokenizer.word_counts)
                         for comment in train_comments]

# %%
test_comments = test['comment_text']
prepro_test_comments = [comment_preprocessing(
    comment, tokenizer.word_counts) for comment in test_comments]


# %%
row = ['id', 'toxic', 'severe_toxic',
       'obscene', 'threat', 'insult', 'identity_hate', 'preprocessed_comment']

train['preprocessed_comment'] = prepro_train_comments
test['preprocessed_comment'] = prepro_test_comments
print(train[:10])
train[row].to_csv('../input/train_prepro.csv', index=False)
test[['id', 'preprocessed_comment']].to_csv(
    '../input/test_prepro.csv', index=False)
