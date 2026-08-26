import numpy as np

def spectral_comb(peaks):
    linear = 10 ** (peaks / 20)

    epsilon = 1e-12
    cepstrum = np.fft.ifft(np.log(linear + epsilon))

    new_mag_spectrum = np.abs(cepstrum)

    new_db_spectrum = 20 * np.log10(new_mag_spectrum + 1e-12)
    return new_db_spectrum