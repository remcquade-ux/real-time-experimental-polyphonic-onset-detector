import numpy as np

def hz_to_cents(freqs, ref_freq=16.3516*2.3516):
    """
    Convert frequencies in Hz to cents relative to a reference (default C1 ≈ 16.16.3516*2 Hz)
    """
    return 1200 * np.log2(freqs / ref_freq)

def spectrum_to_cents(freqs, spectrum, cents_per_semitone=50, ref_freq=16.3516*2):
    """
    Convert linear-FFT spectrum to a uniform cents grid.
    
    Parameters
    ----------
    freqs : np.array
        FFT frequencies in Hz
    spectrum : np.array
        FFT magnitude in dB
    cents_per_semitone : int
        Resolution per semitone (100 = 1 cent per unit)
    ref_freq : float
        Reference frequency for cents calculation
    
    Returns
    -------
    cents_grid : np.array
        Uniformly spaced cents
    spectrum_cents : np.array
        Spectrum mapped to cents bins using min per bin
    """
    # Convert FFT freqs to cents
    freqs_cents = hz_to_cents(freqs, ref_freq)
    
    # Determine total cents range
    total_cents = int(np.ceil(freqs_cents[-1]))
    
    # Create uniform cents grid
    cents_grid = np.arange(0, total_cents + 1, 1)  # 1 cent resolution
    
    # Allocate output
    spectrum_cents = np.full_like(cents_grid, fill_value=np.nan, dtype=float)
    
    # Bin: for each cents grid point, take minimum from all FFT bins that fall into this cent
    start_idx = 0
    for i, c in enumerate(cents_grid):
        # advance start_idx until freqs_cents[start_idx] >= c
        while start_idx < len(freqs_cents) and freqs_cents[start_idx] < c:
            start_idx += 1
        # take all FFT bins in this cent
        end_idx = start_idx
        while end_idx < len(freqs_cents) and freqs_cents[end_idx] < c + 1:
            end_idx += 1
        if end_idx > start_idx:
            # use min to reduce noise
            spectrum_cents[i] = np.min(spectrum[start_idx:end_idx])
    
    # Optional: fill remaining NaNs (if any) with some floor
    spectrum_cents = np.nan_to_num(spectrum_cents, nan=np.min(spectrum))
    
    return cents_grid, spectrum_cents