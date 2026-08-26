import numpy as np
from scipy.signal import find_peaks


def peak_filter(spectrum, freqs,
                          min_width_bins=1,  #should be bin_width
                          min_height_db=-80,  # minimum peak height
                          dump_noise=True,
                          db_floor=-90):
    """
    Detect peaks in an FFT spectrum that are likely harmonics.
    
    Parameters
    ----------
    freqs : np.array
        Frequency bins (Hz)
    magnitude_db : np.array
        Filtered FFT in dB
    min_width_bins : int
        Minimum width in bins for a peak to count
    min_height_db : float
        Minimum amplitude for a peak
    dump_noise : bool
        If True, set all non-peak bins to db_floor
    db_floor : float
        dB value to replace noise bins with
    
    Returns
    -------
    peaks_db : np.array
        Spectrum where only detected harmonic peaks are kept
    peak_freqs : np.array
        Frequencies of detected peaks
    peak_mags : np.array
        Magnitudes of detected peaks
    """

    # Find peaks
    peaks_idx, properties = find_peaks(spectrum, prominence = 30, height=min_height_db,)

    

    peak_freqs = freqs[peaks_idx]
    peak_mags = spectrum[peaks_idx]


    # Start with a clean spectrum
    if dump_noise:
        peaks_db = np.full_like(spectrum, db_floor)
        peaks_db[peaks_idx] = spectrum[peaks_idx]
    else:
        peaks_db = spectrum.copy()

    return peaks_db, peak_freqs, peak_mags