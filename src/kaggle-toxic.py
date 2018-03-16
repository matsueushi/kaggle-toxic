# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.text import text_to_word_sequence, Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Input, Embedding, LSTM, GlobalMaxPool1D, GlobalAveragePooling1D, Dropout, Dense, Bidirectional, concatenate, GRU
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
tfidf_vec = TfidfVectorizer(max_features=50000, min_df=2, max_df=0.5)
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
params = {
    'objective': 'binary:logistic',
    'eta': 0.1,
    'max_depth': 5,
    'silence': 1,
    'eval_metric': 'auc',
    'min_child_weight': 1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
}
submission_xgboost = pd.DataFrame(test['id'])
d_test = xgb.DMatrix(test_dtm)
print(d_test)
for c in LABELS:
    print(c)
    d_train = xgb.DMatrix(train_dtm, label=train[c])
    print(d_train)
    model = xgb.train(params, d_train,  num_boost_round=500)
    prediction = model.predict(d_test)
    print(prediction[:5])
    submission_xgboost[c] = model.predict(d_test)

# %%
# svd = TruncatedSVD(n_components=25, n_iter=30)
# truncated_train = svd.fit_transform(train_dtm)
# truncated_test = svd.fit_transform(test_dtm)


# %%
# print(truncated_train.shape)
# print(truncated_train[0])
# print(train['toxic'].shape)
# print(truncated_test.shape)


# %%
# d_train = xgb.DMatrix(truncated_train, label=train['toxic'])
# d_test = xgb.DMatrix(truncated_test)


# %%
# submission_xgboost = pd.DataFrame(test['id'])
# for c in LABELS:
#     print(c)
#     y_train = np.array(train[c])
#     xgb_classifier = XGBClassifier(
#         learning_rate=0.1,
#         n_estimators=300,
#         max_depth=5,
#         min_child_weight=6,
#         gamma=0.1,
#         subsample=0.6,
#         colsample_bytree=0.9,
#         objective='binary:logistic',
#         scale_pos_weight=1)
#     xgb_classifier.fit(train_array, y_train, eval_metric='auc', verbose=True)
#     proba = xgb_classifier.predict_proba(truncated_test)[:, 1]
#     submission_xgboost[c] = proba
#     print(c + ':finished')


# %%
submission_xgboost.to_csv('./submission_xgboost.csv', index=False)


# %%
# LSTMここから
# tokenizer
NUM_WORDS = 30000
tokenizer = Tokenizer(num_words=NUM_WORDS)
tokenizer.fit_on_texts(train_preprocessed_comment)
tokenized_train = tokenizer.texts_to_sequences(train_preprocessed_comment)
tokenized_test = tokenizer.texts_to_sequences(test_preprocessed_comment)
print(tokenized_train[0])
print(tokenized_test[0])


# %%
MAX_LEN = 150
padded_tokenized_train = pad_sequences(tokenized_train, maxlen=MAX_LEN)
padded_tokenized_test = pad_sequences(tokenized_test, maxlen=MAX_LEN)


# %%
EMBEDDING_SIZE = 200
ip = Input(shape=(MAX_LEN,))
w_count = Input(shape=(1,))  # word_count
x = Embedding(input_dim=NUM_WORDS, output_dim=EMBEDDING_SIZE)(ip)
x = Bidirectional(GRU(units=64, dropout=0.15,
                      recurrent_dropout=0.15, return_sequences=True))(x)
x = concatenate([GlobalMaxPool1D()(x), GlobalAveragePooling1D()(x)])
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
submission_lstm.to_csv('./submission_gru.csv', index=False)

# %%
# LSTMここまで


# %%
# submissionの比較
submission_logistic = pd.read_csv('submission_logistic_reg.csv')
submission_lstm = pd.read_csv('./submission_lstm.csv')
submission_xgboost = pd.read_csv('./submission_xgboost.csv')
print(submission_logistic.head())
print(submission_lstm.head())
print(submission_xgboost.head())


# %%
# モデルの結果の差を見る
std_dev = test[['id', 'preprocessed_comment_text']].copy()
sub_ave = test[['id']].copy()
for c in LABELS:
    sub_ave[c] = (submission_logistic[c] + submission_lstm[c] +
                  submission_xgboost[c]) / 3.0

sub_ave.to_csv('submission_average.csv', index=False)
