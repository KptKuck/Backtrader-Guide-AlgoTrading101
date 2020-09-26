#!/usr/bin/env python
# coding: utf-8


import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import pydot
import graphviz
import tensorflow as tf
from tensorflow.keras.callbacks import *
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import Input, GaussianNoise
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import plot_model


print(tf.__version__)

import datetime
import advance_plotting as ADVplot

LOOKBACK = 12
STEP = 1
FORECAST = 1
INIT_CAPITAL = 10000
STAKE = 10

log_dir = "logs\\fit\\{0}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, min_lr=0.000001, verbose=0)
checkpointer = ModelCheckpoint(filepath="testtest.hdf5", verbose=0, save_best_only=True)
es = EarlyStopping(patience=400)



def create_dataset(data):
    highp = pd.to_numeric(data.iloc[:, 2])
    lowp = pd.to_numeric(data.iloc[:, 3])
    openp = pd.to_numeric(data.iloc[:, 1])
    closep = pd.to_numeric(data.iloc[:, 4])
    # tradesp = pd.to_numeric(data.ix[:, 'Trades'])
    volumep = pd.to_numeric(data.iloc[:, 6])

    normal_close = closep.values.tolist()
    normal_open = openp.values.tolist()

    highp = highp.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
    lowp = lowp.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
    openp = openp.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
    closep = closep.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
    # tradesp = tradesp.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
    volumep = volumep.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()

    X, Y = [], []

    for i in range(0, len(data), STEP):
        try:
            o = openp[i:i + LOOKBACK]
            h = highp[i:i + LOOKBACK]
            l = lowp[i:i + LOOKBACK]
            c = closep[i:i + LOOKBACK]
            v = volumep[i:i + LOOKBACK]
            # t = tradesp[i:i + LOOKBACK]

            y_i = (normal_close[i + LOOKBACK + FORECAST] - normal_open[i + LOOKBACK]) / normal_open[i + LOOKBACK]
            y_i = 1 if y_i > 0 else 0

            x_i = np.column_stack((o, h, l, c, v))

        except Exception as e:
            break

        X.append(x_i)
        Y.append(y_i)

    X, Y = np.array(X), np.array(Y)
    return X, Y






def plot_history(history):
    # summarize history for accuracy
    plt.subplot(2, 1, 1)
    #plt.plot(history.history['accuracy'])
    #plt.plot(history.history['val_accuracy'])
    plt.axhline(y=0.5, color='grey', linestyle='--')
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()
    # summarize history for loss
    plt.subplot(2, 1, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.axhline(y=0.693, color='grey', linestyle='--')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()


def get_lr_model(x1, x2):
    main_input = Input(shape=(x1, x2,), name='main_input')
    x = GaussianNoise(0.01)(main_input)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid", name="out")(x)
    final_model = Model(inputs=[main_input], outputs=[output])
    final_model.compile(optimizer=Adam(lr=0.001, amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
    plot_model(final_model, show_shapes=True, to_file='predict_lstm_autoencoder.png')
    return final_model

def get_lstm_class_model(x1, x2, y1, y2):
    #verbose, epochs, batch_size = 0, 1, 64
    n_timesteps, n_features, n_outputs = x1.shape[1], x1.shape[2], y1.shape[1]

    model = Sequential()
    model.add(LSTM(100, input_shape=(n_timesteps, n_features)))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fit network
    #model.fit(trainX, trainy, epochs=epochs, batch_size=batch_size, verbose=verbose)
    # evaluate model
    #_, accuracy = model.evaluate(testX, testy, batch_size=batch_size, verbose=0)

    return model

def get_lstm_model(x1, x2):
    main_input = Input(shape=(x1, x2,), name='main_input')

    model = Sequential()
    model.add(LSTM(200, activation='relu', kernel_initializer='he_normal', input_shape=(x1, x2)))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')




    #output = Dense(1, activation="sigmoid", name="out")(x)
    #final_model = Model(inputs=[main_input], outputs=[output])
    #final_model.compile(optimizer=Adam(lr=0.001, amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
    #plot_model(model, show_shapes=True, to_file='predict_lstm_autoencoder.png')
    return model


def train_model(model, X_train, Y_train, X_test, Y_test):
    history = model.fit(X_train, Y_train,
                        epochs=1000,
                        batch_size=32,
                        verbose=2,
                        validation_data=(X_test, Y_test),
                        callbacks=[tensorboard_callback, reduce_lr, checkpointer, es],
                        shuffle=True)
    return history
















def sharpe(returns):
    return np.sqrt(len(returns)) * returns.mean() / returns.std()

# print sharpe(our_pct_growth)
# print sharpe(benchmark_ptc_growth)


# returns.tail()
