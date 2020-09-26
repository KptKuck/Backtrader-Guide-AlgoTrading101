import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import advance_plotting as ADV
from scipy.signal import find_peaks
from collections import *

bars = pd.read_csv('TSLA.csv')
x = bars['Close']
xNp = bars.to_numpy()

#fpPar = [namedtuple('Parameter', 'Name, Value')]
fpParD ={
  "distance": "10",
  "prominence": "10",
  "hight": "0"
}
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
