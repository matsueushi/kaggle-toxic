# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import preprocess_string
from keras.preprocessing.text import Tokenizer


# %%
STOPLIST = set(
    'a an the is was are were to for of in at on as with by from and or'.split())


# %%
def comments_preprocessing(comments, word_counts=None, count_threshold=10):
    def is_necessary(s, word_counts):
        ret = (not s.isdigit()) and len(s) > 1  # and s not in STOPLIST
        if word_counts == None:
            return ret
        else:
            if s in word_counts:
                return ret and (word_counts[s] > count_threshold)
            else:
                return False

    def text_preprocessing(text):
        text = text.replace('\'', '')
        seq = preprocess_string(text)
        seq = [s for s in seq if is_necessary(s, word_counts)]
        return seq

    seqs = [text_preprocessing(text) for text in comments]
    texts = [' '.join(s) for s in seqs]
    texts_len = [len(s) for s in seqs]
    return texts, texts_len


# %%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')
train_comments = train['comment_text']
test_comments = test['comment_text']
all_comments = train_comments.append(test_comments)


# %%
tokenizer = Tokenizer()
fit_text, _ = comments_preprocessing(all_comments)
tokenizer.fit_on_texts(fit_text)

train_text, train_len = comments_preprocessing(
    train_comments, tokenizer.word_counts)
test_text, test_len = comments_preprocessing(
    test_comments, tokenizer.word_counts)


# %%
word_series = pd.Series(tokenizer.word_counts)
word_series.sort_values(inplace=True)
word_series[-50:]


# %%
row = ['id', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult',
       'identity_hate', 'preprocessed_comment', 'words']

train['preprocessed_comment'] = train_text
train['words'] = train_len
test['preprocessed_comment'] = test_text
test['words'] = test_len
print(train[:10])
train[row].to_csv('../input/train_prepro.csv', index=False)
test[['id', 'preprocessed_comment', 'words']].to_csv(
    '../input/test_prepro.csv', index=False)
