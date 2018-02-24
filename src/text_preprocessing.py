# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
from gensim.parsing.preprocessing import remove_stopwords
from keras.preprocessing.text import Tokenizer, text_to_word_sequence


# %%
STOPLIST = set(
    'a an the is was are were to for of in at on as with by from and or'.split())


# %%
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# %%
def comment_preprocessing(comment, word_counts=None, count_threshold=10):
    def is_necessary(s, word_counts):
        ret = (not s.isdigit()) and len(s) > 1 and s not in STOPLIST
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
    return seq


# %%
train_comments = train['comment_text']
tokenizer = Tokenizer()
fit_train_comments = [' '.join(comment_preprocessing(comment))
                      for comment in train_comments]
tokenizer.fit_on_texts(fit_train_comments)

# %%
word_series = pd.Series(tokenizer.word_counts)
word_series.sort_values(inplace=True)
word_series[-50:]

# %%
prepro_train_seqs = [comment_preprocessing(comment, tokenizer.word_counts)
                     for comment in train_comments]
prepro_train_comments_number = [len(seq) for seq in prepro_train_seqs]
prepro_train_comments = [' '.join(seq) for seq in prepro_train_seqs]


# %%
test_comments = test['comment_text']
prepro_test_seqs = [comment_preprocessing(
    comment, tokenizer.word_counts) for comment in test_comments]
prepro_test_comments_number = [len(seq) for seq in prepro_test_seqs]
prepro_test_comments = [' '.join(seq) for seq in prepro_test_seqs]


# %%
print(prepro_train_comments[0])
print(prepro_test_comments[0])


# %%
row = ['id', 'toxic', 'severe_toxic',
       'obscene', 'threat', 'insult', 'identity_hate', 'preprocessed_comment', 'words_number']

train['preprocessed_comment'] = prepro_train_comments
train['words_number'] = prepro_train_comments_number
test['preprocessed_comment'] = prepro_test_comments
test['words_number'] = prepro_test_comments_number
print(train[:10])
train[row].to_csv('../input/train_prepro.csv', index=False)
test[['id', 'preprocessed_comment', 'words_number']].to_csv(
    '../input/test_prepro.csv', index=False)
