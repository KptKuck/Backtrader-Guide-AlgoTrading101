import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import advance_plotting as ADV
from scipy.signal import find_peaks
import yfinance as yf
import logging
from UTIL.logger import init_logger
import UTIL.Results as mDir

from collections import *

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

btc = yf.download('BTC-USD', '2019-09-01', interval="1H")

btcNp = btc.to_numpy()
log.info("Read CSV Data")
bars = pd.read_csv('AUDCAD_raw_tick-M5-NoSession.csv', header=0, nrows=20000, parse_dates=[0])

log.info("Dataframe Size: %d", bars.size)
log.info("Dataframe Shape %s", bars.shape.__str__())

xClose = bars['Close']
xNp = bars.to_numpy()

fpPar = {}

fpPar["height"] = None
fpPar["threshold"] = None
fpPar["distance"] = 30
fpPar["prominence"] = 0.005  # 0.001 = 1 Pip
fpPar["width"] = None
fpPar["wlen"] = None
fpPar["rel_height"] = 0.1
fpPar["plateau_size"] = None

log.info("FindPeaks Input Parameter")
for key in fpPar:
    log.info("%s: %s ", key, fpPar[key]  )




def findPeaks(x, par):
    peaksH, propsH = find_peaks(x[:, 2], **par)

    peaksL, propsL = find_peaks(-x[:, 2], **par)

    log.info("peaksH: %s  peaksL: %s ", peaksH.size, peaksL.size)

    return peaksH, peaksL, propsL, propsH


peaksH, peaksL, propsL, propsH = findPeaks(xNp, fpPar)


ADV.plot_bars_peaks(bars, peaksH, peaksL)


