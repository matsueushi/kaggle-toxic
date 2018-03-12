# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, GlobalMaxPool1D, Dropout, Dense, Bidirectional
from keras.models import Model
from gensim.parsing.preprocessing import preprocess_string
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.decomposition import TruncatedSVD
from sklearn.grid_search import GridSearchCV
import xgboost as xgb
from xgboost.sklearn import XGBClassifier


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
# ここまでのテキスト処理を保存
test_save_columns = LABELS[:]
test_save_columns.extend(COMMON_SAVE_COLUMNS)

train[test_save_columns].to_csv('../input/train_prepro.csv')
test[COMMON_SAVE_COLUMNS].to_csv('../input/test_prepro.csv')
train[test_save_columns][:100].to_csv('../input/train_prepro_first100.csv')
test[COMMON_SAVE_COLUMNS][:100].to_csv('../input/test_prepro_first100.csv')


# %%
# (途中から始める場合 状態の読み込み)
train = pd.read_csv('../input/train_prepro.csv')
test = pd.read_csv('../input/test_prepro.csv')
print(train.head())
print(test.head())


# %%
train_preprocessed_comment = train['preprocessed_comment_text'].astype(str)
test_preprocessed_comment = test['preprocessed_comment_text'].astype(str)


# %%
# Logistic Regressionここから
# TfidfVectorizer
tfidf_vec = TfidfVectorizer(max_features=20000, min_df=2, max_df=0.5)
train_dtm = tfidf_vec.fit_transform(train_preprocessed_comment)
test_dtm = tfidf_vec.transform(test_preprocessed_comment)


# %%
print(type(train_dtm))
print(train_dtm[0])


# %%
# logistic regression + grid search
submission_logistic = pd.DataFrame(test['id'])
logistic_reg = LogisticRegression()
param_grid = {'C': [0.01, 0.5, 0.1, 1, 5, 10]}
for c in LABELS:
    train_y = train[c]
    grid_search = GridSearchCV(
        logistic_reg, param_grid=param_grid, scoring='accuracy',)
    grid_search.fit(train_dtm, train_y)
    pred_train_y = grid_search.predict(train_dtm)
    print(c, accuracy_score(train_y, pred_train_y),
          'best param:', grid_search.best_params_)
    pred_test_y = grid_search.predict_proba(test_dtm)[:, 1]
    submission_logistic[c] = pred_test_y


# %%
submission_logistic.to_csv('submission_logistic_reg.csv', index=False)


# %%
# Logistic Regressionここまで


# %%
# XGBoost
# Tfidfを改めて計算(max_featuresは増やす)
tfidf_vec = TfidfVectorizer(max_features=50000, min_df=2, max_df=0.5)
train_dtm = tfidf_vec.fit_transform(train_preprocessed_comment)
test_dtm = tfidf_vec.transform(test_preprocessed_comment)


# %%
svd = TruncatedSVD(n_components=25, n_iter=30)
truncated_train = svd.fit_transform(train_dtm)
truncated_test = svd.fit_transform(test_dtm)


# %%
print(truncated_train.shape)
print(truncated_train[0])
print(train['toxic'].shape)
print(truncated_test.shape)


# %%
d_train = xgb.DMatrix(truncated_train, label=train['toxic'])
d_test = xgb.DMatrix(truncated_test)


# %%
print(d_train)


# %%
train_array = np.array(truncated_train)
label_array = np.array(train['toxic'])


# %%
xgb_classifier = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1)
xgb_param = xgb_classifier.get_xgb_params()
xgb_n_estimators = num_boost_round = xgb_classifier.get_params()[
    'n_estimators']
xgb_cv = xgb.cv(xgb_param, d_train, num_boost_round=xgb_n_estimators,
                nfold=5, metrics='auc', early_stopping_rounds=50, verbose_eval=True)
xgb_classifier.set_params(n_estimators=xgb_cv.shape[0])


# %%
xgb_classifier.fit(train_array, label_array, eval_metric='auc')
proba = xgb_classifier.predict_proba(truncated_test)


# %%
test_param1 = {
    'max_depth': [4, 5, 6],
    'min_child_weight': [4, 5, 6]
}
grid_search1 = GridSearchCV(
    estimator=XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=5,
        min_child_weight=1,
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1),
    param_grid=test_param1,
    scoring='roc_auc',
    iid=False,
    verbose=3,
    cv=3)
grid_search1.fit(train_array, label_array)


# %%
print(grid_search1.grid_scores_)
print(grid_search1.best_params_)
print(grid_search1.best_score_)

# %%
test_param2 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
grid_search2 = GridSearchCV(
    estimator=XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=grid_search1.best_params_['max_depth'],
        min_child_weight=grid_search1.best_params_[
            'min_child_weight'],
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1),
    param_grid=test_param2,
    scoring='roc_auc',
    iid=False,
    verbose=3,
    cv=3)
grid_search2.fit(train_array, label_array)

# %%
test_param2 = {
    'gamma': [i / 10.0 for i in range(0, 5)]
}
grid_search2 = GridSearchCV(
    estimator=XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=grid_search1.best_params_['max_depth'],
        min_child_weight=grid_search1.best_params_[
            'min_child_weight'],
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1),
    param_grid=test_param2,
    scoring='roc_auc',
    iid=False,
    verbose=3,
    cv=3)
grid_search2.fit(train_array, label_array)


# %%
print(grid_search2.grid_scores_)
print(grid_search2.best_params_)
print(grid_search2.best_score_)


# %%
test_param3 = {
    'subsample': [i / 10.0 for i in range(6, 10)],
    'colsample_bytree': [i / 10.0 for i in range(6, 10)]
}
grid_search3 = GridSearchCV(
    estimator=XGBClassifier(
        learning_rate=0.1,
        n_estimators=300,
        max_depth=grid_search1.best_params_['max_depth'],
        min_child_weight=grid_search1.best_params_[
            'min_child_weight'],
        gamma=0,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='binary:logistic',
        nthread=4,
        scale_pos_weight=1),
    param_grid=test_param3,
    scoring='roc_auc',
    iid=False,
    verbose=3,
    cv=3)
grid_search3.fit(train_array, label_array)


# %%
print(grid_search3.grid_scores_)
print(grid_search3.best_params_)
print(grid_search3.best_score_)


# %%
xgb_classifier = XGBClassifier(
    learning_rate=0.1,
    n_estimators=1000,
    max_depth=grid_search1.best_params_['max_depth'],
    min_child_weight=grid_search1.best_params_[
        'min_child_weight'],
    gamma=grid_search2.best_params_['gamma'],
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    nthread=4,
    scale_pos_weight=1)
xgb_classifier.fit(train_array, label_array, eval_metric='auc', verbose=True)
proba = xgb_classifier.predict_proba(truncated_test)


# %%
submission_xgboost = pd.DataFrame({'id': test["id"]})
submission_xgboost['toxic'] = proba[:]
submission_xgboost.to_csv('./submission_xgboost.csv', index=False)


# %%
# LSTMここから
# tokenizer
NUM_WORDS = 20000
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(train_preprocessed_comment)
tokenized_train = tokenizer.texts_to_sequences(train_preprocessed_comment)
tokenized_test = tokenizer.texts_to_sequences(test_preprocessed_comment)
print(tokenized_train[0])
print(tokenized_test[0])


# %%
MAX_LEN = 200
padded_tokenized_train = pad_sequences(tokenized_train, maxlen=MAX_LEN)
padded_tokenized_test = pad_sequences(tokenized_test, maxlen=MAX_LEN)


# %%
EMBEDDING_SIZE = 128
ip = Input(shape=(MAX_LEN,))
w_count = Input(shape=(1,))  # word_count
x = Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_SIZE)(ip)
x = Bidirectional(LSTM(units=64, return_sequences=True))(x)
x = GlobalMaxPool1D()(x)
x = Dropout(0.1)(x)
x = Dense(len(LABELS), activation='sigmoid')(x)
model = Model(inputs=ip, outputs=x)
model.compile(loss='binary_crossentropy',
              optimizer='adam', metrics=['accuracy'])
model.summary()


# %%
train_y = train[LABELS].values
model.fit(padded_tokenized_train, train_y, verbose=1,
          batch_size=64, epochs=3)  # , validation_split=0.1)
prediction = model.predict(padded_tokenized_test, verbose=1, batch_size=64)

# %%
submission_lstm = pd.DataFrame({'id': test["id"]})
for i, c in enumerate(LABELS):
    submission_lstm[c] = prediction[:, i]
submission_lstm.to_csv('./submission_lstm.csv', index=False)

# %%
# LSTMここまで


# %%
# submissionの比較
submission_logistic = pd.read_csv('submission_logistic_reg.csv')
submission_lstm = pd.read_csv('./submission_lstm.csv')
print(submission_logistic.head())
print(submission_lstm.head())


# %%
# モデルの結果の差を見る
std_dev = test[['id', 'preprocessed_comment_text']].copy()
sub_ave = test[['id']].copy()
diff['diff_total'] = 0
for c in LABELS:
    diff['diff_' + c] = submission_logistic[c] - submission_lstm[c]
    diff['diff_total'] += diff['diff_' + c].abs()
    sub_ave[c] = (submission_logistic[c] + submission_lstm[c]) / 2


diff.sort_values(by=['diff_total'], ascending=False, inplace=True)
diff[:100]

sub_ave.to_csv('submission_average.csv', index=False)
