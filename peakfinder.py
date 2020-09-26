import matplotlib.pylab as plt
import numpy as np
import pandas as pd
import advance_plotting as ADV
from scipy.signal import find_peaks

bars = pd.read_csv('TSLA.csv')
x = bars['Close']
xNp = bars.to_numpy()
#new_array = xNp.astype(type('float', (float,), {}))
#yyy = np.invert(new_array)
#xNpDD = np.expand_dims(xNp,axis=0)
#xNpInv = np.linalg.inv(new_array)


def findPeaks(x):
    peaksH, temp1 = find_peaks(x[:,2])

    peaksL, temp2 = find_peaks(-x[:,2])

    return peaksH, peaksL


peaksH, peaksL = findPeaks(xNp)

# ADV.plot_dataset(bars, 'TSLA')
ADV.plot_bars_peaks(bars, peaksH, peaksL)

#plt.plot(x)

#plt.plot(peaks, x[peaks], "x")

#plt.plot(np.zeros_like(x), "--", color="gray")

#plt.show()
