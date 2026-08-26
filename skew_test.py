import numpy as np

def skewness(x):
    x = x - np.mean(x)
    m2 = np.mean(x**2)
    m3 = np.mean(x**3)
    return m3 / (m2**1.5 + 1e-12)


def process_frame(filtered_magnitude, phase, sample_rate):
    # reconstruct
    spectrum = filtered_magnitude * np.exp(1j * phase)
    wave = np.real(np.fft.ifft(spectrum))

    # remove DC
    wave -= np.mean(wave)

    # autocorrelation to find period
    corr = np.correlate(wave, wave, mode='full')
    corr = corr[len(corr)//2:]

    min_period = int(sample_rate / 2000)
    max_period = int(sample_rate / 40)

    period = np.argmax(corr[min_period:max_period]) + min_period

    # first 3 periods
    segment = wave[:period * 3]

    # compute skew
    skew = skewness(segment)

    return skew