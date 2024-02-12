from scipy.io.wavfile import read
from scipy.fftpack import rfft

import numpy as np
import matplotlib.pyplot as plt


rate, data = read("instrument1.wav")

fft_out = rfft(data)

plt.plot(np.abs(fft_out))
plt.show()
