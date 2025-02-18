"""
Course: ENEL420 Advanced Signals
Last Modified on: 15 August 2024
Names: Michael Zhu, Claire Kim
Description:
This script applies an optimal FIR filter using the Remez to remove interference from an audio signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import remez, freqz, lfilter

# Load and normalize the audio signal
fs, data = wavfile.read('groupG.wav')
data = data / np.max(np.abs(data))

# Initialisation
band = [635, 645]
trans_width = 200
Nyquist_freq = fs / 2
bands = [0, band[0] - trans_width, band[0], band[1], band[1] + trans_width, Nyquist_freq]
desired = [1, 0, 1]
weight = [1, 3, 1]  
numtaps = 501
fir_remez = remez(numtaps, bands, desired, weight=weight, fs=fs)

# Apply the FIR filter to the audio signal
filtered_signal_remez = lfilter(fir_remez, 1.0, data)
wavfile.write('optimalFIR.wav', fs, np.int16(filtered_signal_remez * 32767))

# Compute and plot the magnitude spectrum of the filtered signal
fft_filtered = np.abs(np.fft.fft(filtered_signal_remez))
freqs = np.fft.fftfreq(len(fft_filtered), 1/fs)

plt.figure(figsize=(10, 4))
plt.plot(freqs[:len(freqs)//2], fft_filtered[:len(fft_filtered)//2], label='Filtered Signal (remez)')
plt.title('Magnitude Spectrum: Filtered Signal')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude')
plt.legend()
plt.grid()
plt.show()

# Variance calculations
original_variance = np.var(data)
filtered_signal_variance = np.var(filtered_signal_remez)
noise_variance = original_variance - filtered_signal_variance

print(f"Original Signal Var: {original_variance}")
print(f"Filtered Signal Var: {filtered_signal_variance}")
print(f"Noise Var: {noise_variance}")

# Plot the magnitude spectra in dB
magnitude_spectrum_filtered_db = 20 * np.log10(fft_filtered[:len(fft_filtered)//2])

plt.figure(figsize=(10, 4))
plt.plot(freqs[:len(freqs)//2], magnitude_spectrum_filtered_db, label='Filtered Signal (remez)')
plt.title('Magnitude Spectrum: Filtered Signal (in dB)')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Magnitude [dB]')
plt.legend()
plt.grid()
plt.show()

# Plot the phase response of the designed FIR filter
w, h_response = freqz(fir_remez, worN=8000, fs=fs)

plt.figure(figsize=(10, 4))
plt.plot(w, np.angle(h_response), 'b', label='Phase Response')
plt.title('Phase Response of the Designed FIR Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [radians]')
plt.xlim(0, 2000)
plt.grid(True)
plt.show()

# Unwrap the phase response
unwrapped_phase = np.unwrap(np.angle(h_response))

# Plot the unwrapped phase response of the designed FIR filter
plt.figure(figsize=(10, 4))
plt.plot(w, unwrapped_phase, 'b', label='Unwrapped Phase Response')
plt.title('Unwrapped Phase Response of the Designed FIR Filter')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Phase [radians]')
plt.grid(True)
plt.show()