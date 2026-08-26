import numpy as np
from peak_filter import peak_filter

import numpy as np

def noise_filter(spectrum, freqs, DB_MIN=-90):
    filtered = spectrum.copy()
    
    # ... your curvature / floor filtering ...
    filtered[filtered < -75] = DB_MIN
    curvature = np.zeros_like(filtered)
    curvature[1:-1] = filtered[:-2] + filtered[2:] - 2 * filtered[1:-1]
    filtered[abs(curvature) < 10] = DB_MIN ## ASSOCIATED WITH FREQ, WE WILL DRAFT A MAXIMISING FUNCTION LATER. FOR NOW, HIGHER FREQ = BETTER


    # --- Peak detection ---
    peaks_db, peak_freqs, peak_mags = peak_filter(spectrum, freqs)


    return filtered, peak_freqs