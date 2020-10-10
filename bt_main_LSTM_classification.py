import datetime

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import matthews_corrcoef

from strategies import *

import logging
from UTIL.logger import init_logger
import UTIL.Results as mDir

import threading

import advance_plotting as ADVplot

# init logger

log = init_logger(__name__, show_debug=True)
log.info("Logger %s started", __name__)

# check and create results dir
resultPath = ""
res = mDir.check('results', log)
if res[0]:
    print("res ok")
    resultPath = res[2]
    fh = logging.FileHandler(resultPath + "\\client_debug.log")
    fh.setLevel(logging.DEBUG)
    log.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)-15s %(funcName)-30s %(message)s')
    fh.setFormatter(formatter)
    log.addHandler(fh)
else:
    print("res not ok")


# Instantiate Cerebro engine
cerebro = bt.Cerebro()
log.debug("Cerebro instance loaded")
# Set data parameters and add to Cerebro

data = bt.feeds.YahooFinanceCSVData(
    dataname='TSLA.csv',
    fromdate=datetime.datetime(2015, 2, 17),
    todate=datetime.datetime(2018, 1, 1))

log.debug("Backtrader DataFeed loaded: %s", str(data.params.dataname))
log.debug("Feed Parameter: FromDate: %s ", str(data.params.fromdate))
log.debug("Feed Parameter:   ToDate: %s ", str(data.params.todate))
log.debug("Feed Parameter: TimeFrame %s ", str(data.params.timeframe))

data2 = bt.feeds.GenericCSVData(

    dataname="AUDCAD_raw_tick-M5-NoSession.csv",

    fromdate=datetime.datetime(2010, 1, 1),
    todate=datetime.datetime(2012, 1, 1),

    dtformat='%Y-%m-%d %H:%M',

    datetime=0,
    open=1,
    high=2,
    low=3,
    close=4,
    volume=5)

log.debug("Backtrader DataFeed loaded: %s", str(data2.params.dataname))
log.debug("Feed Parameter: FromDate: %s ", str(data2.params.fromdate))
log.debug("Feed Parameter:   ToDate: %s ", str(data2.params.todate))
log.debug("Feed Parameter: TimeFrame %s ", str(data2.params.timeframe))

cerebro.adddata(data)



#bars = pd.read_csv('TSLA.csv')

# th = threading.Thread(target=ADVplot.plot_dataset(bars, 'TSLA'))
# th.start()


cerebro.adddata(data2)

log.info("Read CSV Data")
bars = pd.read_csv('AUDCAD_raw_tick-M5-NoSession.csv', header=0, nrows=500000, parse_dates=[0])

log.info("Dataframe Size: %d", bars.size)
log.info("Dataframe Shape %s", bars.shape.__str__())

xClose = bars['Close']
xNp = bars.to_numpy()

START_TRAIN_DATE = '2010-01-01'
END_TRAIN_DATE = '2010-06-01'
START_TEST_DATE = '2010-06-01'
END_TEST_DATE = '2011-01-01'


log.info("Create train_set and test_set")
train_set = bars[(bars['DateTime'] > START_TRAIN_DATE) & (bars['DateTime'] < END_TRAIN_DATE)]
test_set = bars[(bars['DateTime'] > START_TEST_DATE) & (bars['DateTime'] < END_TEST_DATE)]
log.info("Create ready")

log.info("Create dataset")
X_train, Y_train = create_dataset(train_set)
X_test, Y_test = create_dataset(test_set)
log.info("Create ready")


model = get_lstm_model(X_train.shape[1], X_train.shape[-1])
model.summary()

history = train_model(model, X_train, Y_train, X_test, Y_test)

#mse, mae = model.evaluate(X_test, Y_test, verbose=0)

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

pred = [1 if p == 1 else -1 for p in pred]  # we need to change NN's 0 output to -1 for our strategy
pred = [p if i % FORECAST == 0 else 0 for i, p in enumerate(pred)]
pred = [0.] * LOOKBACK + pred + [
    0] * FORECAST  # first LOOKBACK items needed to make first forecast + items we shifted


class NNclassification(bt.Strategy):

    def log(self, txt, dt=None):
        if not self.printlog:
            return
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))  # Comment this line when running optimiz

    def __init__(self):
        self.close0 = self.datas[0].close
        self.high0 = self.datas[0].high
        self.low0 = self.datas[0].low
        self.open0 = self.datas[0].open
        self.date0 = self.datas[0].datetime
        self.volume0 = self.datas[0].volume

        self.close1 = self.datas[1].close
        self.high1 = self.datas[1].high
        self.low1 = self.datas[1].low
        self.open1 = self.datas[1].open
        self.date1 = self.datas[1].datetime
        self.volume1 = self.datas[1].volume

        self.signal_last = None
        self.printlog = True
        self.signal_changed = False
        self.signal = 0
        self.pred = 0
        self.NN_input = 0
        self.bt_pd = 0
        self.bar_executed = 0

        # ohlbar = [self.data1.lines[i][0] for i in range(data.size())]
        # self.pred = model.predict(self.datas[:])
        # self.signal = [1 if p > 0.5 else 0 for p in self.pred]
        # self.C = confusion_matrix(Y_test, self.signal)

        # Order variable will contain ongoing order details/status
        self.order = None

    def __bt_to_numpy__(self, maxlen):
        total = []
        for i in range(-maxlen, 1):
            o = self.open0[i]
            h = self.high0[i]
            c = self.close0[i]
            l = self.low0[i]
            v = self.volume0[i]

            data_t = np.column_stack((o, h, l, c, v))
            total.append(data_t)

        total = np.array(total)
        return total

    def calc_NN_input(self):

        dlist = self.bt_pd.tolist()

        highp = pd.Series((i[0][0] for i in dlist))
        lowp = pd.Series((i[0][1] for i in dlist))
        openp = pd.Series((i[0][2] for i in dlist))
        closep = pd.Series((i[0][3] for i in dlist))
        volumep = pd.Series((i[0][4] for i in dlist))

        # normal_close = closep.values.tolist()
        # normal_open = openp.values.tolist()

        highp = highp.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
        lowp = lowp.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
        openp = openp.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
        closep = closep.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
        # tradesp = tradesp.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()
        volumep = volumep.pct_change().replace(np.nan, 0).replace(np.inf, 0).values.tolist()

        X, Y = [], []

        for i in range(0, len(self.bt_pd), STEP):
            try:
                o = openp[i:i + LOOKBACK]
                h = highp[i:i + LOOKBACK]
                l = lowp[i:i + LOOKBACK]
                c = closep[i:i + LOOKBACK]
                v = volumep[i:i + LOOKBACK]
                # t = tradesp[i:i + LOOKBACK]

                # y_i = (normal_close[i + LOOKBACK + FORECAST] - normal_open[i + LOOKBACK]) / normal_open[i + LOOKBACK]
                # y_i = 1 if y_i > 0 else 0

                x_i = np.column_stack((o, h, l, c, v))
            except Exception as e:
                print(e)
                break

            X.append(x_i)

        X = np.array(X[:93])
        return X

    def notify_order(self, order):
        self.log('Active Order Status: , %s' % order.getstatusname())

        if order.status in [order.Submitted, order.Accepted]:
            #An active Buy/Sell order has been submitted/accepted - Nothing to do
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('BUY EXECUTED, %.2f' % order.executed.price)
        elif order.issell():
            self.log('SELL EXECUTED, %.2f' % order.executed.price)
            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        self.order = None

    def next(self):
        # Generate Signal
        self.bt_pd = self.__bt_to_numpy__(130)
        self.NN_input = self.calc_NN_input()

        # self.log('Close: %.2f, ATR: %.4f' % (self.dataclose[0], ATR))
        self.pred = model.predict(self.NN_input, batch_size=64)
        self.signal = [1 if p > 0.5 else 0 for p in self.pred]
        # self.log('Signal: %.d, ' % (self.signal[0]))
        # self.signal_last = self.signal[0]

        if self.signal_last is None:
            self.log('First Bar Return ')
            self.signal_last = self.signal[0]
            return

        if self.signal[0] == self.signal_last:
            self.signal_changed = False
            self.signal_last = self.signal[0]

        elif self.signal[0] != self.signal_last:
            self.signal_changed = True
            # self.log('Signal changed %.2f' % self.close0[0])
            self.signal_last = self.signal[0]

        # Check for open orders
        if self.order:
            return

        if not self.position:
            # We are not in the market, look for a signal to OPEN trades

            if self.signal_changed and self.signal[0] == 1:
                # self.log('BUY CREATE, %.2f' % self.close0[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()
            # Otherwise
            elif self.signal_changed and self.signal[0] == 0:
                # self.log('SELL CREATE, %.2f' % self.close0[0])
                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()
        else:
            # We are already in the market, look for a signal to CLOSE trades
            # if len(self) >= (self.bar_executed + 5):
            #    self.log('CLOSE CREATE, %.2f' % self.dataclose[0])
            #    self.order = self.close()
            if self.signal_changed:
                # self.log('CLOSE CREATE, %.2f' % self.close0[0])
                self.order = self.close()


# Add strategy to Cerebro
cerebro.addstrategy(NNclassification)

cerebro.addanalyzer(bt.analyzers.DrawDown, _name='drawdown')

# Default position size
cerebro.addsizer(bt.sizers.SizerFix, stake=3)

start_portfolio_value = cerebro.broker.getvalue()
# Run Cerebro Engine
results = cerebro.run(maxcpus=1)
pl = cerebro.plot()

end_portfolio_value = cerebro.broker.getvalue()
pnl = end_portfolio_value - start_portfolio_value
print('Starting Portfolio Value: %.2f' % start_portfolio_value)
print('Final Portfolio Value: %.2f' % end_portfolio_value)
print('PnL: %.2f' % pnl)
