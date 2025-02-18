"""
Course: ENEL420 Advanced Signals
Last Modified on: 15 August 2024
Names: Michael Zhu, Claire Kim
Description:
This script applies a frequency sampling FIR filter to remove noise from an audio signal.
used firwin2 to design the filter
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin2, freqz, lfilter

fs, data = wavfile.read('groupG.wav')
data = data / (2**15)

noise_freq_low, noise_freq_high = 600, 680
cutoff_low = noise_freq_low / (0.5 * fs)
cutoff_high = noise_freq_high / (0.5 * fs)
numtaps = 2047
freqs = [0, cutoff_low-0.01, cutoff_low, cutoff_high, cutoff_high+0.01, 1]
gains = [1, 1, 0, 0, 1, 1]
fir_freq_sampling = firwin2(numtaps, freqs, gains)

w, h = freqz(fir_freq_sampling, worN=8000)
plt.figure(figsize=(10, 4))
plt.plot(0.5*fs*w/np.pi, np.abs(h), 'b', label='Frequency Sampling FIR Filter')
plt.title('Frequency Response of the Frequency Sampling FIR Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.legend()
plt.grid()
plt.show()

filtered_signal_freq_sampling = lfilter(fir_freq_sampling, 1.0, data)
wavfile.write('frequencySamplingFIR.wav', fs, (filtered_signal_freq_sampling * (2**15)).astype(np.int16))

fft_filtered = np.fft.fft(filtered_signal_freq_sampling)
fft_freqs = np.fft.fftfreq(len(fft_filtered), 1/fs)

plt.figure(figsize=(10, 4))
plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_filtered)[:len(fft_filtered)//2])
plt.title('Magnitude Spectrum of the Filtered Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.show()


magnitude_spectrum_db = 20 * np.log10(np.abs(fft_filtered)[:len(fft_filtered)//2])

plt.figure(figsize=(10, 4))
plt.plot(fft_freqs[:len(fft_freqs)//2], magnitude_spectrum_db)
plt.title('Magnitude Spectrum of the Filtered Signal (in dB)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.grid()
plt.show()

time = np.arange(len(data)) / fs
plt.figure(figsize=(10, 4))
plt.plot(time, data, label='Original Signal')
plt.plot(time, filtered_signal_freq_sampling, label='Filtered Signal')
plt.title('Time-Domain Signal (Original and Filtered)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

