# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# %%
# データの読み込み
train = pd.read_csv('../input/train.csv')
test = pd.read_csv('../input/test.csv')


# %%
train.head()


# %%
test.head()


# %%
LABELS = ['toxic', 'severe_toxic',
          'obscene', 'threat', 'insult', 'identity_hate']
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
    df['word_count'] = df['comment_text'].apply(
        lambda x: len(text_to_word_sequence(x)))
    print(df['word_count'].head())


# %%
fig, ax = plt.subplots(nrows=len(LABELS),
                       sharex=True, sharey=True, figsize=(10, 20))
for i, c in enumerate(LABELS):
    not_toxic = train[train[c] == 0]
    toxic = train[train[c] == 1]
    print(len(not_toxic), len(toxic))
    ax[i].hist(not_toxic['word_count'], bins=250,
               normed=1, alpha=0.5, label='not_toxic')
    ax[i].hist(toxic['word_count'], bins=250,
               normed=1, alpha=0.5, label='toxic')
    ax[i].set_title(c)
    ax[i].legend()
plt.xlim(0, 200)


# %%
for df in [train, test]:
    comments = df['comment_text']
    preprocessed_comment_seqs = [preprocess_string(c) for c in comments]
    df['preprocessed_comment_text'] = [
        ' '.join(s) for s in preprocessed_comment_seqs]
    df['preprocessed_word_count'] = [len(s) for s in preprocessed_comment_seqs]


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
COMMON_SAVE_COLUMNS = ['id', 'preprocessed_comment_text',
                       'word_count', 'preprocessed_word_count']
test_save_columns = LABELS[:]
test_save_columns.extend(COMMON_SAVE_COLUMNS)

train[test_save_columns].to_csv('../input/train_prepro.csv')
test[COMMON_SAVE_COLUMNS].to_csv('../input/test_prepro.csv')
train[test_save_columns][:100].to_csv('../input/train_prepro_first100.csv')
test[COMMON_SAVE_COLUMNS][:100].to_csv('../input/test_prepro_first100.csv')


# %%
# TfidfVectorizer
tfidf_vec = TfidfVectorizer(max_features=20000, min_df=2)
train_dtm = tfidf_vec.fit_transform(train['preprocessed_comment_text'])
test_dtm = tfidf_vec.transform(test['preprocessed_comment_text'])


# %%
print(type(train_dtm))
print(train_dtm[0])


# %%
submission = pd.DataFrame(test['id'])
logistic_reg = LogisticRegression(C=5.0)
for c in LABELS:
    train_y = train[c]
    logistic_reg.fit(train_dtm, train_y)
    pred_train_y = logistic_reg.predict(train_dtm)
    print(c, accuracy_score(train_y, pred_train_y))
    pred_test_y = logistic_reg.predict_proba(test_dtm)[:, 1]
    submission[c] = pred_test_y


# %%


# %%
submission.to_csv('submission_logistic_reg.csv', index=False)
