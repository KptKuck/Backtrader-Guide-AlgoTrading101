import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import advance_plotting as ADV
from scipy.signal import find_peaks
import yfinance as yf

from collections import *

btc = yf.download('BTC-USD', '2019-09-01',  interval="1H")

btcNp = btc.to_numpy()
bars = pd.read_csv('AUDCAD-M1-Forex_245.csv')

xClose = bars['Close']
xNp = bars.to_numpy()



fpPar = {}

fpPar["height"] = None
fpPar["threshold"] = None
fpPar["distance"] = None
fpPar["prominence"] = 30
fpPar["width"] = None
fpPar["wlen"] = None
fpPar["rel_height"] = 0.5
fpPar["plateau_size"] = None





def findPeaks(x,  par):
    peaksH, propsH = find_peaks(x[:, 2], **par)

    peaksL, propsL = find_peaks(-x[:, 2], **par)

    return peaksH, peaksL, propsL, propsH


peaksH, peaksL, propsL, propsH = findPeaks(xNp, fpPar)

# ADV.plot_dataset(bars, 'TSLA')
ADV.plot_bars_peaks(bars, peaksH, peaksL)

# plt.plot(x)

# plt.plot(peaks, x[peaks], "x")

# plt.plot(np.zeros_like(x), "--", color="gray")

# plt.show()
