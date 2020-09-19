import datetime

from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import matthews_corrcoef

from LSTM_Classification import *
from strategies import *

# Instantiate Cerebro engine
cerebro = bt.Cerebro()

# Set data parameters and add to Cerebro
data = bt.feeds.YahooFinanceCSVData(
    dataname='TSLA.csv',
    fromdate=datetime.datetime(2016, 1, 1),
    todate=datetime.datetime(2016, 12, 31))

cerebro.adddata(data)

START_TRAIN_DATE: str = '2015-01-01'
END_TRAIN_DATE = '2017-12-31'
START_TEST_DATE = '2018-01-01'
END_TEST_DATE = '2018-03-09'



bars = pd.read_csv('TSLA.csv')

train_set = bars[(bars['Date'] > START_TRAIN_DATE) & (bars['Date'] < END_TRAIN_DATE)]

test_set = bars[(bars['Date'] > START_TEST_DATE) & (bars['Date'] < END_TEST_DATE)]

X_train, Y_train = create_dataset(train_set)
X_test, Y_test = create_dataset(test_set)

model = get_lr_model(X_train.shape[1], X_train.shape[-1])
model.summary()

history = model.fit(X_train, Y_train,
              epochs = 1000,
              batch_size = 64,
              verbose=0,
              validation_data=(X_test, Y_test),
              callbacks=[reduce_lr, checkpointer, es],
              shuffle=True)

# plot_history(history)
pred = model.predict(X_test)

pred = [1 if p > 0.5 else 0 for p in pred]
C = confusion_matrix(Y_test, pred)

print('MATTHEWS CORRELATION')
print(matthews_corrcoef(Y_test, pred))
print('CONFUSION MATRIX')
print(C / C.astype(np.float).sum(axis=1))
print('CLASSIFICATION REPORT')
print(classification_report(Y_test, pred))
print('-' * 20)

# Add strategy to Cerebro
cerebro.addstrategy(NNclassification)

# Run Cerebro Engine
cerebro.run()
