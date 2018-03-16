# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV


# %%
COMMON_SAVE_COLUMNS = ['id', 'preprocessed_comment_text',
                       'word_count', 'preprocessed_word_count']
LABELS = ['toxic', 'severe_toxic',
          'obscene', 'threat', 'insult', 'identity_hate']


# %%
# データの読み込み
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# %%
train.head()


# %%
test.head()


# %%
ind = np.arange(len(LABELS))
plt.style.use('bmh')
fix, ax = plt.subplots()
sum_labels = [train[l].sum() for l in LABELS]
plot_bars = plt.bar(ind, sum_labels)
ax.set_title('Toxicity types')
ax.yaxis.grid()
ax.set_xticks(ind)
ax.set_xticklabels(LABELS)


# %%
for df in [train, test]:
    df['raw_word_count'] = df['comment_text'].apply(
        lambda x: len(text_to_word_sequence(x)))
    print(df['raw_word_count'].head())


# %%
def plot_count(df, col_name, bins, lim=200):
    fig, ax = plt.subplots(nrows=len(LABELS),
                           sharex=True, sharey=True, figsize=(10, 20))
    for i, c in enumerate(LABELS):
        not_toxic = df[train[c] == 0]
        toxic = df[train[c] == 1]
        print(len(not_toxic), len(toxic))
        ax[i].hist(not_toxic[col_name].dropna(), bins=bins,
                   normed=1, alpha=0.5, label='not_toxic')
        ax[i].hist(toxic[col_name].dropna(), bins=bins,
                   normed=1, alpha=0.5, label='toxic')
        ax[i].set_title(c)
        ax[i].legend()
    plt.xlim(0, lim)


# %%
plot_count(train, 'raw_word_count', 250)


# %%
for df in [train, test]:
    comments = df['comment_text']
    preprocessed_comment_seqs = [preprocess_string(c) for c in comments]
    df['preprocessed_comment_text'] = [
        ' '.join(s) for s in preprocessed_comment_seqs]
    df['word_count'] = [len(s) for s in preprocessed_comment_seqs]
    df['unique_word_count'] = [len(set(s)) for s in preprocessed_comment_seqs]
    df['duplication'] = df['word_count'] - df['unique_word_count']
    df['duplication_rate'] = df['duplication'] / df['word_count']

# %%
plot_count(train, 'word_count', 200)


# %%
plot_count(train, 'unique_word_count', 200, 100)


# %%
print(train['duplication_rate'])
plot_count(train, 'duplication_rate', 30, 1.0)


# %%
# 単語の頻度測定


def get_word_list(comments, csv_path):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(comments)
    word_list = pd.Series(tokenizer.word_counts)
    word_list.sort_values(ascending=False, inplace=True)
    word_list.to_csv(csv_path)
    return word_list


all_word_list = get_word_list(
    train['preprocessed_comment_text'], '../input/all_word_list.csv')

not_toxic_word_list = {}
toxic_word_list = {}
for c in LABELS:
    not_toxic = train[train[c] == 0]
    toxic = train[train[c] == 1]
    not_toxic_word_list[c] = get_word_list(not_toxic['preprocessed_comment_text'],
                                           '../input/' + c + '_positive_word_list.csv')
    toxic_word_list[c] = get_word_list(toxic['preprocessed_comment_text'],
                                       '../input/' + c + '_negative_word_list.csv')

# %%
plt.figure(figsize=(15, 4))
plt.title('Top 50 word frequencies(All)')
all_word_list[:50].plot.bar()
for c in LABELS:
    plt.figure(figsize=(15, 4))
    plt.title('Top 50 word frequencies(Class=' + c + ', Positive)')
    not_toxic_word_list[c][:50].plot.bar()
    plt.figure(figsize=(15, 4))
    plt.title('Top 50 word frequencies(Class=' + c + ', Negative)')
    toxic_word_list[c][:50].plot.bar()


# %%
for c in LABELS:
    not_toxic = train[train[c] == 0]
    toxic = train[train[c] == 1]
    not_tokenizer = Tokenizer()
    not_toxic_word_list[c] = not_tokenizer.fit_on_texts(
        not_toxic['preprocessed_comment_text'])
    not_dic = pd.Series(not_tokenizer.word_docs)
    not_dic /= len(not_toxic)
    print(not_dic.sort_values())
    tokenizer = Tokenizer()
    toxic_word_list[c] = tokenizer.fit_on_texts(
        toxic['preprocessed_comment_text'])
    dic = pd.Series(tokenizer.word_docs)
    dic /= len(toxic)
    print(dic.sort_values())


# %%
train['preprocessed_comment_text'].head()


# %%
test['preprocessed_comment_text'].head()


# %%
# ここまでのテキスト処理を保存
test_save_columns = LABELS[:]
test_save_columns.extend(COMMON_SAVE_COLUMNS)

train[test_save_columns].to_csv('../input/train_prepro.csv')
test[COMMON_SAVE_COLUMNS].to_csv('../input/test_prepro.csv')
train[test_save_columns][:100].to_csv('../input/train_prepro_first100.csv')
test[COMMON_SAVE_COLUMNS][:100].to_csv('../input/test_prepro_first100.csv')
