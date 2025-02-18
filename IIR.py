"""
Course: ENEL420 Advanced Signals
Last Modified on: 15 August 2024
Names: Michael Zhu, Claire Kim 
Description:
This script applies an IIR filter to remove interference from an audio signal.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import lfilter
from scipy.signal import freqz

def sampled_time_domain_plot(time, data):
    plt.plot(time, data)
    plt.title('Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def sampled_magnitude_plot(freq, fft_spectrum):
    plt.plot(freq, 20 * np.log10(fft_spectrum))
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)
    plt.show()
  

def IIR_filter(fft_spectrum, freq, fs, data, time):
    #Parameters
    #interference signal is at approximately 635-645 Hz
    interference_freq = 642.09168 #640.2028661264966
    bandwidth = 2500
    r = 1-(bandwidth/fs) * np.pi  
    print(r) 
    theta = (2 * np.pi * interference_freq) / fs 
    b1 = -2 * np.cos(theta)
    a1 = -2 * r * np.cos(theta)
    a2 = r**2

    b = [1, b1, 1]
    a = [1, a1, a2]
    filtered_data = lfilter(b, a, data)
    #amplify filtered data by 10 so audio file is louder 
    filtered_data_int = np.int16(filtered_data * (2**15)) * 10
    wavfile.write('IIR.wav', fs, filtered_data_int)
    w, h = freqz(b, a, worN=8000)

    return a, b, w, h, filtered_data


def filtered_magnitude_plot(w, h, fs):
    plt.subplot(2, 1, 1)
    plt.plot(w * fs / (2 * np.pi), 20 * np.log10(np.abs(h)))
    plt.title('Magnitude Response of IIR Filter')
    #plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.grid(True)

def filtered_phase_plot(w, h, fs): 
    plt.subplot(2, 1, 2)
    plt.plot(w * fs / (2 * np.pi), np.angle(h))
    plt.title('Phase Response of IIR Filter')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Phase [radians]')
    plt.grid(True)
    plt.show()


def filtered_time_domain_plot(time, filtered_data):
    plt.plot(time, filtered_data)
    plt.title('Filtered Time Domain')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.show()

def filtered_spectrum_plot(filtered_data, freq):
    fft_spectrum_filtered = np.fft.fft(filtered_data)
    plt.plot(freq, fft_spectrum_filtered)
    plt.title('Magnitude Spectrum Filtered')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude')
    plt.grid(True)
    plt.show()

def pole_placement_plot(zeros, poles):
    plt.plot(np.real(zeros), np.imag(zeros), 'go', label='Zeros')
    plt.plot(np.real(poles), np.imag(poles), 'rx', label='Poles')
    plt.legend()

    unit_circle = plt.Circle((0, 0), 1, color='blue', fill=False, linestyle='--')
    plt.gca().add_artist(unit_circle)

    plt.xlim([-1.5, 1.5])
    plt.ylim([-1.5, 1.5])
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.title('Pole-Zero Plot')
    plt.xlabel('Real')
    plt.ylabel('Imaginary')
    plt.grid(True)
    plt.show()

def plot_input_vs_output_magnitude_db(freq, fft_spectrum, fft_spectrum_filtered):
    plt.plot(freq, 20 * np.log10(fft_spectrum), label='Input Signal', color='blue')
    plt.plot(freq, 20 * np.log10(fft_spectrum_filtered), label='Filtered Signal', color='orange')
    plt.title('Magnitude Spectrum (Input vs Filtered)')
    plt.xlabel('Frequency [Hz]')
    plt.ylabel('Magnitude [dB]')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    fs, data = wavfile.read('groupG.wav')
    data=data/(2**15)

    time = np.arange(0, data.size/ fs, 1/ fs) 

    fft_spectrum = np.abs(np.fft.fft(data))
    freq = np.fft.fftfreq(data.size, 1 / fs)

    a, b, w, h, filtered_data = IIR_filter(fft_spectrum, freq, fs, data, time)

    zeros = np.roots(b)
    poles = np.roots(a)

    
    fft_spectrum_filtered = np.abs(np.fft.fft(filtered_data))
    #plots
    sampled_time_domain_plot(time, data)
    sampled_magnitude_plot(freq, fft_spectrum)
    pole_placement_plot(zeros, poles)
    filtered_magnitude_plot(w, h, fs)
    filtered_phase_plot(w, h, fs)
    filtered_time_domain_plot(time, filtered_data)
    filtered_spectrum_plot(filtered_data, freq)
    plot_input_vs_output_magnitude_db(freq, fft_spectrum, fft_spectrum_filtered)
main()


