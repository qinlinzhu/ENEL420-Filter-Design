"""
Course: ENEL420 Advanced Signals
Last Modified on: 15 August 2024
Names: Michael Zhu, Claire Kim
Description: 
This script designs and applies a custom Hamming window FIR filter to remove interference from an audio signal. 
"""

import numpy as np
from scipy.io import wavfile
from scipy.signal import lfilter, freqz
import matplotlib.pyplot as plt


Fs, data = wavfile.read('groupG.wav')
data = data / (2**15)

start = 1200  # Start of notch in Hz
stop = 1400   # End of notch in Hz
N = 2007      # Num of filter coefficients


window = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(N) / (N - 1))
wc1 = (start / Fs) * np.pi
wc2 = (stop / Fs) * np.pi
n = (N - 1) / 2
hd = np.zeros(N)

for i in range(N):
    if i == n:
        hd[i] = 1 - ((wc2 - wc1) / np.pi)
    else:
        hd[i] = ((np.sin(np.pi * (i - n)) - (np.sin(wc2 * (i - n)) - np.sin(wc1 * (i - n)))) / (np.pi * (i - n)))


h = hd * window
h /= np.sum(h)
w, h_response = freqz(h, worN=8000)
plt.figure(figsize=(10, 4))
plt.plot(0.5 * Fs * w / np.pi, np.abs(h_response), 'b', label='Designed FIR Filter')
plt.title('Frequency Response of the homemade Hamming window FIR Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Gain')
plt.grid()
plt.show()

filtered_data = lfilter(h, 1.0, data)
filtered_data_int = np.int16(filtered_data * (2**15))
wavfile.write('homemadeWindowFIR.wav', Fs, filtered_data_int)

def plot_magnitude_spectrum(signal, Fs, label):
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(spectrum), 1/Fs)
    plt.plot(freqs[:len(freqs)//2], 20 * np.log10(np.abs(spectrum[:len(spectrum)//2])), label=label)

plt.figure(figsize=(10, 4))
plot_magnitude_spectrum(data, Fs, 'Original Signal')
plot_magnitude_spectrum(filtered_data, Fs, 'Filtered Signal')
plt.title('Magnitude Spectrum of Original and Filtered Signals')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude (dB)')
plt.legend()
plt.grid(True)
plt.show()

original_variance = np.var(data)
filtered_signal_variance = np.var(filtered_data)
noise_variance = original_variance - filtered_signal_variance
print(f"Original Signal Var: {original_variance}")
print(f"Filtered Signal Var: {filtered_signal_variance}")
print(f"Noise Var: {noise_variance}")

plt.figure(figsize=(10, 4))
plt.plot(0.5 * Fs * w / np.pi, np.angle(h_response),  label='Phase Response')
plt.title('Phase Response of the Designed FIR Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [radians]')
plt.grid(True)
plt.xlim(0, 2000)  
plt.show()

def plot_magnitude_spectrum(signal, Fs, label):
    spectrum = np.fft.fft(signal)
    freqs = np.fft.fftfreq(len(spectrum), 1/Fs)
    plt.plot(freqs[:len(freqs)//2], np.abs(spectrum[:len(spectrum)//2]), label=label)

plt.figure(figsize=(10, 4))
plot_magnitude_spectrum(data, Fs, 'Original Signal')
plot_magnitude_spectrum(filtered_data, Fs, 'Filtered Signal')
plt.title('Magnitude Spectrum of Original and Filtered Signals (Linear Scale)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid(True)
plt.show()
