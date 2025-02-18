"""
Course: ENEL420 Advanced Signals
Last Modified on: 15 August 2024
Names: Michael Zhu, Claire Kim
Description:
This script designs and applies a Hamming window FIR filter to remove noise from an audio signal.
used firwin to design the filter.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import firwin, freqz, lfilter

fs, data = wavfile.read('groupG.wav')
data = data / (2**15)

cutoff_low = 600 / (0.5 * fs)
cutoff_high = 680 / (0.5 * fs)
numtaps = 2047
fir_hamming = firwin(numtaps, [cutoff_low, cutoff_high], pass_zero='bandstop', window='hamming')

w, h = freqz(fir_hamming, worN=8000)
plt.figure(figsize=(10, 4))
plt.plot(0.5 * fs * w / np.pi, np.abs(h), 'b')
plt.title('Frequency Response of the Hamming FIR Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.show()

filtered_signal_hamming = lfilter(fir_hamming, 1.0, data)

wavfile.write('hammingWindowFIR.wav', fs, (filtered_signal_hamming * (2**15)).astype(np.int16))

time = np.arange(len(data)) / fs
plt.figure(figsize=(10, 4))
plt.plot(time, data, label='Original Signal')
plt.plot(time, filtered_signal_hamming, label='Filtered Signal', linestyle='--')
plt.title('Time-Domain Signal (Original and Filtered)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.legend()
plt.grid()
plt.show()

fft_filtered = np.fft.fft(filtered_signal_hamming)
fft_freqs = np.fft.fftfreq(len(fft_filtered), 1/fs)

plt.figure(figsize=(10, 4))
plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_filtered)[:len(fft_filtered)//2])
plt.title('Magnitude Spectrum of the Filtered Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.show()

original_variance = np.var(data)
filtered_signal_variance = np.var(filtered_signal_hamming)
noise_variance = original_variance - filtered_signal_variance

print(f"Original Signal Var: {original_variance}")
print(f"Filtered Signal Var: {filtered_signal_variance}")
print(f"Noise Var: {noise_variance}")

fft_original = np.fft.fft(data)
magnitude_spectrum_filtered_db = 20 * np.log10(np.abs(fft_filtered)[:len(fft_filtered)//2])
magnitude_spectrum_original_db = 20 * np.log10(np.abs(fft_original)[:len(fft_original)//2])

plt.figure(figsize=(10, 4))
plt.plot(fft_freqs[:len(fft_freqs)//2], magnitude_spectrum_original_db, label='Original Signal')
plt.plot(fft_freqs[:len(fft_freqs)//2], magnitude_spectrum_filtered_db, label='Filtered Signal')
plt.title('Magnitude Spectrum (dB)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10, 4))
plt.plot(0.5 * fs * w / np.pi, np.angle(h), 'b')
plt.title('Phase Response of the FIR Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase (radians)')
plt.xlim(0, 500)
plt.grid()
plt.show()
