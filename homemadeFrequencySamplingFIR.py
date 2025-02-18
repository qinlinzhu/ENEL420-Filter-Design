"""
Course: ENEL420 Advanced Signals
Last Modified on: 15 August 2024
Names: Michael Zhu, Claire Kim
Description:
This script applies a homemade frequency sampling FIR filter to attenuate noise in an audio signal.
"""

import numpy as np
from scipy.io import wavfile
from scipy.fft import ifft
from scipy.signal import lfilter
import matplotlib.pyplot as plt

# Load and normalize the audio signal
samplerate, data = wavfile.read('groupG.wav')
data = data / (2**15)

# Initialise filter parameters
N = 345
Fs = samplerate
f_start, f_stop = 635, 645
freqs = np.fft.fftfreq(N, d=1/Fs)

H = np.ones(N)  # Start with an all-pass filter
k_start = int(f_start / Fs * N)
k_stop = int(f_stop / Fs * N)
H[k_start:k_stop+1] = 0  # Notch filter
H = np.concatenate([H[:N//2], H[:N//2][::-1]])  # Symmetry for real impulse response

h_manual_freq_sampling = np.real(ifft(H))
h_manual_freq_sampling /= np.sum(h_manual_freq_sampling)

# Apply the filter to the audio data
filtered_data = lfilter(h_manual_freq_sampling, 1.0, data)
wavfile.write('homemadeFrequencySamplingFIR.wav', Fs, np.int16(filtered_data * (2**15)))

# Plot the filter's gain and phase response
plt.figure(figsize=(10, 6))
plt.plot(freqs[:N//2], np.abs(H[:N//2]))
plt.title('Gain vs Frequency of the homemade Frequency Sampling FIR Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Gain')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(freqs[:N//2], np.angle(H[:N//2]))
plt.title('Phase Response of the homemade Frequency Sampling FIR Filter')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Phase [radians]')
plt.grid(True)
plt.show()

# Plot time-domain signals (original and filtered)
time_vector = np.arange(len(data)) / Fs
plt.figure(figsize=(10, 4))
plt.plot(time_vector, data, label='Original Signal')
plt.plot(time_vector, filtered_data, label='Filtered Signal')
plt.title('Time-Domain Signal: Amplitude vs. Time')
plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)
plt.show()

# Calculate and plot the magnitude spectrum
fft_filtered = np.fft.fft(filtered_data)
fft_freqs = np.fft.fftfreq(len(fft_filtered), 1/samplerate)

plt.figure(figsize=(10, 4))
plt.plot(fft_freqs[:len(fft_freqs)//2], np.abs(fft_filtered)[:len(fft_filtered)//2])
plt.title('Magnitude Spectrum of the Filtered Signal using Frequency Sampling Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.grid()
plt.show()

# Variance calculations
original_variance = np.var(data)
filtered_signal_variance = np.var(filtered_data)
noise_variance = original_variance - filtered_signal_variance

print(f"Original Signal Var: {original_variance}")
print(f"Filtered Signal Var: {filtered_signal_variance}")
print(f"Noise Var: {noise_variance}")

# Plot the magnitude spectrum in dB for both original and filtered signals
fft_original = np.fft.fft(data)
magnitude_spectrum_filtered_db = 20 * np.log10(np.abs(fft_filtered)[:len(fft_filtered)//2])
magnitude_spectrum_original_db = 20 * np.log10(np.abs(fft_original)[:len(fft_original)//2])

plt.figure(figsize=(10, 4))
plt.plot(fft_freqs[:len(fft_freqs)//2], magnitude_spectrum_original_db, label='Original Signal')
plt.plot(fft_freqs[:len(fft_freqs)//2], magnitude_spectrum_filtered_db, label='Filtered Signal')
plt.title('Magnitude Spectrum of the Original and Filtered Signals (in dB)')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude [dB]')
plt.legend()
plt.grid()
plt.show()
