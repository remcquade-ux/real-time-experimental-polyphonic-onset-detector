import numpy as np


def reconstruct_wave_rfft(filtered_magnitude, phase):
    spectrum = filtered_magnitude * np.exp(1j * phase)
    wave = np.fft.irfft(spectrum)
    return wave