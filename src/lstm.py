# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Adam
from keras.layers import Input, LSTM, GlobalMaxPool1D, Dense
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model

from gensim.models import KeyedVectors

# %%
FILTERS = 60
EMBEDDING_DIM = 300
STOPLIST = set(
    'a an the is are to for of in at on as with by from and or'.split())
LIST_CLASSES = ["toxic", "severe_toxic",
                "obscene", "threat", "insult", "identity_hate"]


# %%
# 単語が必要か否か
def if_necessary_word(tokenizer, word):
    # STOPLISTに入っておらず、2回以上出現しているもの
    if word in tokenizer.word_counts:
        return (word not in STOPLIST) and (tokenizer.word_counts[word] > 1)
    else:
        return False


# %%
def get_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    print('Found %s unique tokens.' % len(tokenizer.word_index))
    return tokenizer


def texts_to_sequences(tokenizer, texts,  kind=None, maxlen=100):
    if kind == 'filtered':
        # いらない単語をフィルタリングしてトークン化
        tokens = [[tokenizer.word_index[word] for word in text_to_word_sequence(
            text) if if_necessary_word(tokenizer, word)] for text in texts]
    else:
        # 単語に何も条件を付けないでトークン化
        tokens = tokenizer.texts_to_sequences(texts)
    return pad_sequences(tokens, maxlen=maxlen)


# %%
def get_lstm_model(embedding_layer, seq_length, filters, embedding_dim):
    inputs = Input(shape=(seq_length,))
    embedding = embedding_layer(inputs)
    lstm = LSTM(filters, return_sequences=True)(embedding)
    maxpooling = GlobalMaxPool1D()(lstm)
    dense = Dense(50, activation="relu")(maxpooling)
    outputs = Dense(units=len(LIST_CLASSES), activation='sigmoid')(dense)
    model = Model(inputs, outputs)
    return model


def get_callbacks():
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    fpath = './weights.hdf5'
    mcp = ModelCheckpoint(filepath=fpath, monitor='var_loss',
                          verbose=1, save_best_only=True, mode='auto')
    return [es, mcp]


# %%
# ファイルの読み込みと表示
train = pd.read_csv('../input/train.csv')
print(train.head())
train_comment_text = train['comment_text']
tokenizer = get_tokenizer(train_comment_text)
x_train = texts_to_sequences(tokenizer, train_comment_text, 'filtered')
y_train = train[LIST_CLASSES].values

test = pd.read_csv('../input/test.csv')
print(test.head())
test_comment_text = test['comment_text']
x_test = texts_to_sequences(tokenizer, test_comment_text, 'filtered')


# %%
x_train[10]


# %%
word2vec_model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=500000)
embedding_layer = word2vec_model.get_keras_embedding()


# %%
seq_length = x_train.shape[1]
keras_model = get_lstm_model(
    embedding_layer, seq_length, FILTERS, EMBEDDING_DIM)
adam = Adam(lr=1e-3)
keras_model.compile(loss='binary_crossentropy',
                    optimizer=adam, metrics=['acc'])
keras_model.fit(x_train, y_train, epochs=3, verbose=1,
                callbacks=get_callbacks(), batch_size=1000, validation_split=0.2)

prediction = keras_model.predict(x_test, verbose=1, batch_size=1000)

# %%
submission = pd.DataFrame({'id': test["id"]})
for i, x in enumerate(LIST_CLASSES):
    submission[x] = prediction[:, i]
submission.to_csv('./submission.csv', index=False)
