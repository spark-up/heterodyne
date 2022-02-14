# Copyright 2020 Vraj Shah, Arun Kumar
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
from typing import List, Tuple

import keras
import keras.optimizers
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from keras.callbacks import ModelCheckpoint
from keras.layers import (
    BatchNormalization,
    Conv1D,
    Dense,
    Dropout,
    Embedding,
    GlobalMaxPool1D,
    Input,
    concatenate,
)
from keras.models import Model, load_model
from keras.preprocessing import sequence as keras_seq
from keras.preprocessing import text as keras_text
from keras.utils.np_utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import StandardScaler

from ..common import DICT_LABELS
from ..common import abs_limit_10000 as abs_limit
from ..common import to_string_list
from ..lazy_resources import load_test, load_train

# define network parameters
MAX_FEATURES = 256
MAXLEN = 256


def prepare_data(df: pd.DataFrame, y: pd.DataFrame):
    df = df[
        [
            'total_vals',
            'num_nans',
            '%_nans',
            'num_of_dist_val',
            '%_dist_val',
            'mean',
            'std_dev',
            'min_val',
            'max_val',
            'mean_word_count',
            'std_dev_word_count',
            'mean_stopword_total',
            'mean_whitespace_count',
            'mean_char_count',
            'mean_delim_count',
            'stdev_stopword_total',
            'stdev_whitespace_count',
            'stdev_char_count',
            'stdev_delim_count',
        ]
    ]

    df = df.reset_index(drop=True)
    df = df.fillna(0)

    df = df.rename(
        columns={
            'mean': 'scaled_mean',
            'std_dev': 'scaled_std_dev',
            'min_val': 'scaled_min',
            'max_val': 'scaled_max',
            'mean_word_count': 'scaled_mean_token_count',
            'std_dev_word_count': 'scaled_std_dev_token_count',
            '%_nans': 'scaled_perc_nans',
            'mean_stopword_total': 'scaled_mean_stopword_total',
            'mean_whitespace_count': 'scaled_mean_whitespace_count',
            'mean_char_count': 'scaled_mean_char_count',
            'mean_delim_count': 'scaled_mean_delim_count',
            'stdev_stopword_total': 'scaled_stdev_stopword_total',
            'stdev_whitespace_count': 'scaled_stdev_whitespace_count',
            'stdev_char_count': 'scaled_stdev_char_count',
            'stdev_delim_count': 'scaled_stdev_delim_count',
        }
    )

    cols_to_abs_limit = [
        'num_nans',
        'num_of_dist_val',
        'scaled_max',
        'scaled_mean',
        'scaled_min',
        'scaled_std_dev',
        'total_vals',
    ]
    for col in cols_to_abs_limit:
        df[col] = df[col].apply(abs_limit)

    cols_to_normalize = [
        'total_vals',
        'num_nans',
        'num_of_dist_val',
        'scaled_mean',
        'scaled_std_dev',
        'scaled_min',
        'scaled_max',
    ]
    X = df[cols_to_normalize].values
    X = np.nan_to_num(X)
    X_scaled = StandardScaler().fit_transform(X)
    df[cols_to_normalize] = pd.DataFrame(
        X_scaled,
        columns=cols_to_normalize,
        index=df.index,
    )

    y['y_act'] = y['y_act'].map(DICT_LABELS).astype(float)

    return df, y


# %%


def to_padded_sequences(tokenizer: keras_text.Tokenizer, texts: List[str]):
    return keras_seq.pad_sequences(
        tokenizer.texts_to_sequences(texts), maxlen=MAXLEN
    )


X_train = load_train()
X_test = load_test()


def process_data(
    X_: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    atr: pd.Series[str] = X_['Attribute_name']
    samp: pd.Series[str] = X_['sample_1']
    y = X_[['y_act']]

    X_, y = prepare_data(X_, y)
    X_.reset_index(inplace=True, drop=True)
    y.reset_index(inplace=True, drop=True)
    structured_data = X_

    sentences = to_string_list(atr.values)
    sample_sentences = to_string_list(samp.values)

    return X_, y


# for i in range(0,1000,10):
X_train = X_train.sample(frac=1, random_state=100).reset_index(drop=True)
# print(len(xtrain))

atr_train = X_train.loc[:, ['Attribute_name']]
atr_test = X_test.loc[:, ['Attribute_name']]
# print(atr_train)

samp_train = X_train.loc[:, ['sample_1']]
samp_test = X_test.loc[:, ['sample_1']]

y_train = X_train.loc[:, ['y_act']]
y_test = X_test.loc[:, ['y_act']]


X_train, y_train = prepare_data(X_train, y_train)
X_test, y_test = prepare_data(X_test, y_test)

X_train.reset_index(inplace=True, drop=True)
y_train.reset_index(inplace=True, drop=True)
X_test.reset_index(inplace=True, drop=True)
y_test.reset_index(inplace=True, drop=True)


X_train = X_train.values
y_train = y_train.values

X_test = X_test.values
y_test = y_test.values


structured_data_train = X_train
structured_data_test = X_test


sentences_train = to_string_list(atr_train['Attribute_name'].values)
sentences_test = to_string_list(atr_test['Attribute_name'].values)

sample_sentences_train = to_string_list(samp_train['sample_1'].values)
sample_sentences_test = to_string_list(samp_test['sample_1'].values)

tokenizer = keras_text.Tokenizer(char_level=True)
tokenizer.fit_on_texts(list(sentences_train))

tokenizer_sample = keras_text.Tokenizer(char_level=True)
tokenizer_sample.fit_on_texts(list(sample_sentences_train))

# train data
X_train = to_padded_sequences(tokenizer, sentences_train)
X_sample_train = to_padded_sequences(tokenizer, sample_sentences_train)

# test data
X_test = to_padded_sequences(tokenizer_sample, sentences_test)
X_sample_test = to_padded_sequences(tokenizer_sample, sample_sentences_test)


# %%
def build_model(neurons, numfilters, embed_size):
    name_input = Input(shape=(None,))
    x = Embedding(
        input_dim=len(tokenizer.word_counts) + 1,
        output_dim=embed_size,
    )(name_input)
    out_conv = []

    for _ in range(2):
        x = Conv1D(
            numfilters,
            kernel_size=3,
            activation='tanh',
            kernel_initializer='glorot_normal',
        )(x)
        numfilters = numfilters * 2

    out_conv += [GlobalMaxPool1D()(x)]
    out_conv += [GlobalMaxPool1D()(x)]
    x += [GlobalMaxPool1D()(x)]
    word_xy = concatenate(out_conv, axis=-1)

    sample_input = Input(shape=(None,))
    x = Embedding(
        input_dim=len(tokenizer.word_counts) + 1,
        output_dim=embed_size,
    )(sample_input)
    out_conv = []

    for _ in range(2):
        x = Conv1D(
            numfilters,
            kernel_size=3,
            activation='tanh',
            kernel_initializer='glorot_normal',
        )(x)
        numfilters = numfilters * 2

    out_conv += [GlobalMaxPool1D()(x)]
    out_conv += [GlobalMaxPool1D()(x)]
    x += [GlobalMaxPool1D()(x)]
    sample_xy = concatenate(out_conv, axis=-1)

    structured_input = Input(shape=(19,))
    layers_final = concatenate([word_xy, sample_xy, structured_input])
    x = BatchNormalization()(layers_final)

    x = Dense(neurons, activation='tanh')(x)
    x = Dropout(0.5)(x)
    x = Dense(neurons, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(9, activation='softmax')(x)
    model = Model(
        inputs=[name_input, sample_input, structured_input],
        outputs=[x],
    )
    model.compile(
        loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']
    )
    return model


model = build_model(100, 100, 100)
model.summary()


# %%
y_train = y_train.values
structured_data_train = structured_data_train.values

# %%
BATCH_SIZE = 128
EPOCHS = os.getenv('EPOCHS', 25)

K = 5
kf = KFold(n_splits=K)

NEURONS = [100, 500, 1000]
N_FILTERS_GRID = [32, 64, 128]
EMBED_SIZE = [64, 128, 256]

history = None
best_model = None

models = []

avgsc_lst, avgsc_val_lst, avgsc_train_lst = [], [], []
avgsc = avgsc_val = avgsc_train = 0
i = 0
for train_index, test_index in kf.split(X_train):
    file_path = 'CNN_best_model%d.h5' % i

    checkpoint = ModelCheckpoint(
        file_path,
        monitor='val_accuracy',
        verbose=1,
        save_best_only=True,
        mode='max',
    )

    callbacks_list = [checkpoint]  # early

    X_train_cur, X_test_cur = X_train[train_index], X_train[test_index]
    X_train_cur1, X_test_cur1 = (
        X_sample_train[train_index],
        X_sample_train[test_index],
    )
    y_train_cur, y_test_cur = y_train[train_index], y_train[test_index]
    structured_data_train_cur, structured_data_test_cur = (
        structured_data_train[train_index],
        structured_data_train[test_index],
    )

    X_train_train, X_val, y_train_train, y_val = train_test_split(
        X_train_cur, y_train_cur, test_size=0.25, random_state=100
    )
    structured_data_train_train, structured_data_val = train_test_split(
        structured_data_train_cur, test_size=0.25, random_state=100
    )

    bestscore = 0
    for neuro in NEURONS:
        for ne in N_FILTERS_GRID:
            for md in EMBED_SIZE:
                print('\n-------------\n')
                print('Neurons:' + str(neuro))
                print(
                    'Num Filters:' + str(ne) + '   ' + 'Embed Size:' + str(md)
                )
                clf = build_model(neuro, ne, md)
                history = clf.fit(
                    [X_train_cur, X_train_cur1, structured_data_train_cur],
                    to_categorical(y_train_cur),
                    validation_data=(
                        [X_test_cur, X_test_cur1, structured_data_test_cur],
                        to_categorical(y_test_cur),
                    ),
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    shuffle=True,
                    callbacks=callbacks_list,
                )

                best_model = load_model('CNN_best_model' + str(i) + '.h5')

                loss, bscr_train = best_model.evaluate(
                    [X_train_cur, X_train_cur1, structured_data_train_cur],
                    to_categorical(y_train_cur),
                )
                print(loss, bscr_train)
                loss, bscr_val = best_model.evaluate(
                    [X_test_cur, X_test_cur1, structured_data_test_cur],
                    to_categorical(y_test_cur),
                )
                print(loss, bscr_val)
                loss, bscr = best_model.evaluate(
                    [X_test, X_sample_test, structured_data_test],
                    to_categorical(y_test),
                )
                print(loss, bscr)
                print('\n-------------\n')

    best_model = load_model('CNN_best_model' + str(i) + '.h5')

    loss, bscr_train = best_model.evaluate(
        [X_train_cur, X_train_cur1, structured_data_train_cur],
        to_categorical(y_train_cur),
    )
    print(loss, bscr_train)
    loss, bscr_val = best_model.evaluate(
        [X_test_cur, X_test_cur1, structured_data_test_cur],
        to_categorical(y_test_cur),
    )
    print(loss, bscr_val)
    loss, bscr = best_model.evaluate(
        [X_test, X_sample_test, structured_data_test], to_categorical(y_test)
    )
    print(loss, bscr)

    models.append(clf)

    avgsc_train = avgsc_train + bscr_train
    avgsc_val = avgsc_val + bscr_val
    avgsc = avgsc + bscr

    avgsc_train_lst.append(bscr_train)
    avgsc_val_lst.append(bscr_val)
    avgsc_lst.append(bscr)

    print('The training accuracy is:')
    print(bscr_train)
    print('The validation accuracy is:')
    print(bscr_val)
    print('The test accuracy is:')
    print(bscr)
    print('\n')
    i = i + 1


# %%


print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %%
kf = KFold(n_splits=5)
avgsc_lst, avgsc_val_lst, avgsc_train_lst = [], [], []

i = 0
for train_index, test_index in kf.split(X_train):
    X_train_cur, X_test_cur = X_train[train_index], X_train[test_index]
    X_train_cur1, X_test_cur1 = (
        X_sample_train[train_index],
        X_sample_train[test_index],
    )
    y_train_cur, y_test_cur = y_train[train_index], y_train[test_index]
    print(len(X_train_cur), len(X_test_cur))
    print(len(y_train_cur), len(y_test_cur))
    structured_data_train_cur, structured_data_test_cur = (
        structured_data_train[train_index],
        structured_data_train[test_index],
    )
    #     print(len(structured_data_train_cur),len(structured_data_test_cur))
    print(len(X_test), len(y_test))

    best_model = load_model('CNN_best_model' + str(i) + '.h5')

    loss, bscr_train = best_model.evaluate(
        [X_train_cur, X_train_cur1, structured_data_train_cur],
        to_categorical(y_train_cur),
    )
    print(loss, bscr_train)
    loss, bscr_val = best_model.evaluate(
        [X_test_cur, X_test_cur1, structured_data_test_cur],
        to_categorical(y_test_cur),
    )
    print(loss, bscr_val)
    loss, bscr = best_model.evaluate(
        [X_test, X_sample_test, structured_data_test], to_categorical(y_test)
    )
    print(loss, bscr)

    avgsc_train_lst.append(bscr_train)
    avgsc_val_lst.append(bscr_val)
    avgsc_lst.append(bscr)
    print('\n')
    i = i + 1
print(avgsc_train_lst)
print(avgsc_val_lst)
print(avgsc_lst)


# %%
print(avgsc_train_lst)
print(avgsc_val_lst)
print(avgsc_lst)
print(np.mean(avgsc_train_lst))
print(np.mean(avgsc_val_lst))
print(np.mean(avgsc_lst))

y_pred = best_model.predict([X_test, X_sample_test, structured_data_test])
y_pred1 = [np.argmax(i) for i in y_pred]
cm = confusion_matrix(y_test, y_pred1)
print('Confusion Matrix: Actual (Row) vs Predicted (Column)')
print(cm)
