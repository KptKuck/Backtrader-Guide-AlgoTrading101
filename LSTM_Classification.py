#!/usr/bin/env python
# coding: utf-8


import matplotlib.pylab as plt
import numpy as np
import pandas as pd
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Input, GaussianNoise
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

# symbol = 'LTC1518'
# bars = pd.read_csv('./data/%s.csv' % symbol, header=0, parse_dates=['Date'])


# START_TRAIN_DATE = '2015-01-01'
# END_TRAIN_DATE = '2017-12-31'
# START_TEST_DATE = '2018-01-01'
# END_TEST_DATE = '2018-03-09'
LOOKBACK = 7
STEP = 1
FORECAST = 1
INIT_CAPITAL = 10000
STAKE = 10


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
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
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
    return final_model


def get_lstm_model(x1, x2):
    main_input = Input(shape=(x1, x2,), name='main_input')
    x = GaussianNoise(0.01)(main_input)
    x = Flatten()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    output = Dense(1, activation="sigmoid", name="out")(x)
    final_model = Model(inputs=[main_input], outputs=[output])
    final_model.compile(optimizer=Adam(lr=0.001, amsgrad=True), loss='binary_crossentropy', metrics=['accuracy'])
    return final_model


def train_model(model, X_train, Y_train, X_test, Y_test):
    history = model.fit(X_train, Y_train,
                        epochs=1000,
                        batch_size=64,
                        verbose=0,
                        validation_data=(X_test, Y_test),
                        callbacks=[reduce_lr, checkpointer, es],
                        shuffle=True)


reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=50, min_lr=0.000001, verbose=0)
checkpointer = ModelCheckpoint(filepath="testtest.hdf5", verbose=0, save_best_only=True)
es = EarlyStopping(patience=100)


# model = get_lr_model(X_train.shape[1], X_train.shape[-1])
# model = get_model(X_train.shape[1], X_train.shape[-1])
# model.summary()


# model.load_weights('testtest.hdf5')
# pred = model.predict(X_test)
# plot_history(history)

# pred = [1 if p > 0.5 else 0 for p in pred]
# C = confusion_matrix(Y_test, pred)
#
# print 'MATTHEWS CORRELATION'
# print matthews_corrcoef(Y_test, pred)
# print 'CONFUSION MATRIX'
# print(C / C.astype(np.float).sum(axis=1))
# print 'CLASSIFICATION REPORT'
# print classification_report(Y_test, pred)
# print '-' * 20


# In[ ]:


# pred = [1 if p == 1 else -1 for p in pred]  # we need to change NN's 0 output to -1 for our strategy
# pred = [p if i % FORECAST == 0 else 0 for i, p in enumerate(pred)]
# pred = [0.] * (LOOKBACK) + pred + [
#     0] * FORECAST  # first LOOKBACK items needed to make first forecast + items we shifted
#


# class MachineLearningForecastingStrategy(Strategy):
#
#     def __init__(self, symbol, bars, pred):
#         self.symbol = symbol
#         self.bars = bars
#
#     def generate_signals(self):
#         signals = pd.DataFrame(index=self.bars.index)
#         signals['signal'] = pred
#         return signals


# preparing for forecasting for tomorrow!
# test_set['Close'] = test_set['Close'].shift(-FORECAST)
#
# rfs = MachineLearningForecastingStrategy('LTC', test_set, pred)
# signals = rfs.generate_signals()
# portfolio = MarketIntradayPortfolio('LTC', test_set, signals, INIT_CAPITAL, STAKE)
# returns = portfolio.backtest_portfolio()


# returns['signal'] = signals
# our_pct_growth = returns['total'].pct_change().cumsum()
# benchmark_ptc_growth = test_set['Close'].pct_change().cumsum()


# plt.figure()
# plt.plot(returns['total'])
# plt.show()


# plt.figure()
# plt.plot(our_pct_growth, label = 'ML long/short strategy', linewidth=2)
# plt.plot(benchmark_ptc_growth, linestyle = '--', label = 'Buy and hold strategy', linewidth=2)
# plt.legend()
# plt.show()


def sharpe(returns):
    return np.sqrt(len(returns)) * returns.mean() / returns.std()

# print sharpe(our_pct_growth)
# print sharpe(benchmark_ptc_growth)


# returns.tail()
