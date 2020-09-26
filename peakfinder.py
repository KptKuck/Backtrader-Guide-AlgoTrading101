import matplotlib.pyplot as plt
import numpy as np
from scipy.misc import electrocardiogram

from scipy.signal import find_peaks

x = electrocardiogram()[2000:4000]

peaks, _ = find_peaks(x, height=0)

plt.plot(x)

plt.plot(peaks, x[peaks], "x")

plt.plot(np.zeros_like(x), "--", color="gray")

plt.show()