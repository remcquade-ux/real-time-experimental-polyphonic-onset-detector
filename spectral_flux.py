import numpy as np

def detect_note_change(current_mag, prev_mag, 
                                     threshold=0.8, 
                                     min_flux=0.15,
                                     print_details=False):
    """
    Computes spectral flux between two consecutive unfiltered magnitude spectra
    and decides if there was a note change / onset.
    
    Parameters:
    -----------
    current_mag : np.ndarray
        Magnitude spectrum of the current frame (linear scale, not dB).
        Should be the full FFT magnitude before any peak filtering.
    prev_mag : np.ndarray
        Magnitude spectrum of the previous frame (same length and scale).
    threshold : float
        Main decision threshold for normalized spectral flux (tune this).
        Higher = fewer false positives, lower = more sensitive.
    min_flux : float
        Minimum raw flux value to even consider an onset (helps ignore silence).
    print_details : bool
        If True, prints extra debug info (flux value, etc.).
    
    Prints:
    -------
    "Note change / onset detected!" or "No note change."
    """
    if current_mag.shape != prev_mag.shape:
        raise ValueError("current_mag and prev_mag must have the same shape")
    
    # Standard spectral flux: sum of positive differences (half-wave rectified)
    diff = current_mag - prev_mag
    flux = np.sum(np.maximum(diff, 0.0))   # only positive changes matter
    
    # Optional: normalize by the total energy of previous frame (makes it more robust)
    prev_energy = np.sum(prev_mag) + 1e-12
    normalized_flux = flux / prev_energy
    
    # Decision logic
    onset_detected = flux > min_flux and normalized_flux > threshold

    if onset_detected and print_details:
        print(f"Onset | raw flux: {flux:.4f} | normalized: {normalized_flux:.4f}")

    return onset_detected


# -------------------------------------------------
# Simple usage example (for testing)
# -------------------------------------------------
if __name__ == "__main__":
    # Simulate two frames (replace with your real FFT magnitudes)
    N = 1024
    freq_bins = np.arange(N)
    
    # Previous frame: steady note (e.g. C2 + harmonics)
    prev_mag = np.zeros(N)
    prev_mag[::50] = np.array([1.0, 0.7, 0.5, 0.4, 0.3, 0.25, 0.2])  # mock harmonics
    
    # Current frame: new note starts (sudden change)
    current_mag = np.zeros(N)
    current_mag[::48] = np.array([1.0, 0.75, 0.55, 0.45, 0.35, 0.28, 0.22])  # slightly different
    
  
    detect_note_change(current_mag, prev_mag, threshold=0.75, print_details=True)
    
  
    detect_note_change(prev_mag, prev_mag, threshold=0.75, print_details=True)