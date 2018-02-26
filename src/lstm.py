# -*- coding: utf-8 -*-
# %%
import pandas as pd
import numpy as np

import h5py

from keras.preprocessing.text import Tokenizer, text_to_word_sequence
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.optimizers import Adam


from keras.layers import Input, LSTM, GlobalMaxPool1D, Dense, Dropout, Embedding
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import Model

from gensim.models import KeyedVectors

# %%
word2vec_model = KeyedVectors.load_word2vec_format(
    'GoogleNews-vectors-negative300.bin', binary=True, limit=1000000)

# %%
FILTERS = 60
MAXLEN = 100
MAX_FEAUTURE = 50000
DROPOUT_RATE = 0.1
DENSE_UNITS = 50
EMBED_SIZE = 128
LIST_CLASSES = ["toxic", "severe_toxic",
                "obscene", "threat", "insult", "identity_hate"]


# %%
def get_tokenizer(texts):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    print('Found %s unique tokens.' % len(tokenizer.word_index))
    return tokenizer


# %%
def get_lstm_model(embedding_layer, filters):
    inputs = Input(shape=(MAXLEN,))
    embedding = embedding_layer(inputs)
    lstm = LSTM(filters, return_sequences=True)(embedding)
    maxpooling = GlobalMaxPool1D()(lstm)
    dropout_maxpooling = Dropout(DROPOUT_RATE)(maxpooling)
    dense = Dense(DENSE_UNITS, activation="relu")(dropout_maxpooling)
    dropout_dense = Dropout(DROPOUT_RATE)(dense)
    outputs = Dense(units=len(LIST_CLASSES),
                    activation='sigmoid')(dropout_dense)
    model = Model(inputs, outputs)
    model.summary()
    return model


def get_callbacks():
    es = EarlyStopping(monitor='val_loss', patience=3, verbose=1, mode='auto')
    fpath = './lstm_earlystopping_weights.hdf5'
    mcp = ModelCheckpoint(filepath=fpath, monitor='var_loss',
                          verbose=1, save_best_only=True, mode='auto')
    return [es, mcp]


# %%
# ファイルの読み込みと表示
train = pd.read_csv('../input/train_prepro.csv')
test = pd.read_csv('../input/test_prepro.csv')
print(train.head())
print(test.head())
train_comment_text = train['preprocessed_comment'].astype(str)
test_comment_text = test['preprocessed_comment'].astype(str)
all_comment_text = train_comment_text.append(test_comment_text)

tokenizer = Tokenizer()
tokenizer.fit_on_texts(all_comment_text)
x_train = pad_sequences(tokenizer.texts_to_sequences(
    train_comment_text), maxlen=MAXLEN)
y_train = train[LIST_CLASSES].values
x_test = pad_sequences(tokenizer.texts_to_sequences(
    test_comment_text), maxlen=MAXLEN)


# %%
x_train[7]


# %%
embedding_layer = Embedding(MAX_FEAUTURE, EMBED_SIZE)

# %%
embedding_layer = word2vec_model.get_keras_embedding()


# %%
keras_model = get_lstm_model(embedding_layer, FILTERS)
adam = Adam(lr=1e-3)
keras_model.compile(loss='binary_crossentropy',
                    optimizer=adam, metrics=['acc'])


# %%
print(x_train.shape)
print(y_train.shape)
keras_model.fit(x_train, y_train, epochs=10, verbose=1,
                callbacks=get_callbacks(), batch_size=1024)

# %%
keras_model.save("lstm_model_pretrained.hdf5")
keras_model.save_weights("lstm_model_weights_pretrained.hdf5")

# %%
keras_model.load_weights("lstm_model_weights.hdf5")

# %%
keras_model.fit(x_train, y_train, epochs=15, verbose=1, initial_epoch=10,
                callbacks=get_callbacks(), batch_size=1024, validation_split=0.1)

#%%
prediction = keras_model.predict(x_test, verbose=1, batch_size=1024)


# %%
submission = pd.DataFrame({'id': test["id"]})
for i, x in enumerate(LIST_CLASSES):
    submission[x] = prediction[:, i]
submission.to_csv('./lstm_submission.csv', index=False)
