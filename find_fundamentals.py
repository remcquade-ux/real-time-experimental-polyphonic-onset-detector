import numpy as np


# test_freqs = np.array([65.41, 65.41*2+2, 65.41*3+3, 65.41*4+6, 65.41*5+10, 65.41*6+15, 65.41*7+20,
#                        73.42, 73.42*2, 73.42*3, 73.42*4, 73.42*5, 73.42*6, 73.42*7,
#                        100, 102, 896, 322, 30])


def clean_peaks(peak_freqs, cents_tol):
    """
    Merge peaks that are within ±cents_tol (in cents).
    
    Returns:
        cleaned: array of cleaned peak frequencies
        cluster_sizes: how many peaks were merged into each cleaned peak
    """
    if len(peak_freqs) == 0:
        return np.array([]), np.array([])

    log_peaks = np.log2(peak_freqs)
    tol = cents_tol / 1200  # 5 cents in log2 space

    order = np.argsort(log_peaks)
    log_sorted = log_peaks[order]
    peaks_sorted = peak_freqs[order]

    cleaned = []
    cluster_sizes = []

    i = 0
    N = len(log_sorted)

    while i < N:
        cluster = [peaks_sorted[i]]
        j = i + 1
        while j < N and (log_sorted[j] - log_sorted[i]) < tol:
            cluster.append(peaks_sorted[j])
            j += 1

        cleaned.append(np.mean(cluster))
        cluster_sizes.append(len(cluster))
        i = j

    return np.array(cleaned), np.array(cluster_sizes)

def cluster_candidates(candidates, cents_tol):
    """
    Cluster candidate fundamental frequencies using a tolerance in cents.
    
    Args:
        candidates (array-like): Candidate frequencies (Hz).
        cents_tol (float): Cluster tolerance in cents.

    Returns:
        clusters_mean (np.ndarray): Mean frequency of each cluster.
        clusters_size (np.ndarray): Number of candidates in each cluster.
    """
    if len(candidates) == 0:
        return np.array([]), np.array([])

    # Convert candidates to log2 scale to measure cents
    log_candidates = np.log2(candidates)
    tol = cents_tol / 1200  # 1 octave = 1200 cents

    # Sort candidates in log2 space
    order = np.argsort(log_candidates)
    log_sorted = log_candidates[order]
    cand_sorted = candidates[order]

    clusters_mean = []
    clusters_size = []

    i = 0
    N = len(log_sorted)

    while i < N:
        cluster_vals = [cand_sorted[i]]
        j = i + 1

        while j < N and (log_sorted[j] - log_sorted[i]) < tol:
            cluster_vals.append(cand_sorted[j])
            j += 1

        # Store mean and size
        clusters_mean.append(np.mean(cluster_vals))
        clusters_size.append(len(cluster_vals))

        i = j

    return np.array(clusters_mean), np.array(clusters_size)





def find_candidates(peak_freqs,
                          f_min=60,
                          f_max=5000,
                          cents_tol=50,
                          ratio_threshold=0.2):



    if len(peak_freqs) == 0:
        return np.array([])

    peak_freqs, _ = clean_peaks(peak_freqs, cents_tol)

    # --------------------------------
    # STEP 1 — generate candidates
    # --------------------------------

    candidates = []

    for fp in peak_freqs:

        n = 1

        while True:

            f0 = fp / n

            if f0 < f_min or n > 20:
                break

            candidates.append(f0)

            n += 1

    if len(candidates) == 0:
        return np.array([])

    candidates = np.array(candidates)


    # --------------------------------
    # STEP 2 — cluster candidates
    # --------------------------------

    means, sizes = cluster_candidates(candidates, cents_tol)



    # --------------------------------
    # STEP 3 — harmonic tally
    # --------------------------------

    detected = []

## we want this to be a different cent_tol bc harmonic deviation is a bitch +- 30 cents 

    tol_ratio = 2 ** (cents_tol / 1200)

    for idx, f0 in enumerate(means):
        cluster_votes = sizes[idx]  # number of peaks in this fundamental cluster

        multiples = peak_freqs / f0
        # check which multiples are within ±tol_ratio of an integer
        nearest_int = np.round(multiples)
        deviation = multiples / nearest_int
        valid_mask = (deviation >= 1/tol_ratio) & (deviation <= tol_ratio)

        if not np.any(valid_mask):
            continue

        ## pick the largest harmonic number observed
        #f_max_candidate = min(np.max(nearest_int[valid_mask]) * f0, 20*f0)

        #expected = int(np.floor(f_max_candidate / f0))
        #if expected == 0:
        #    continue

        #ratio = cluster_votes / expected
        ##print(f"F0 candidate: {f0:.1f} Hz, votes: {cluster_votes}, expected: {expected}, ratio: {ratio:.2f}")
        #if ratio >= ratio_threshold and cluster_votes > 6:
            #detected.append(f0)
    #for f in detected:
        #print(f"{f:.1f} Hz")
    #print("--------------------------------------------------------------------------")
    #return np.array(detected)
    return means
#detected =find_fundamentals(test_freqs)

#print("Detected fundamentals:", detected)


import numpy as np

def detect_fundamentals(peak_freqs, candidate_f0s,
                        cents_tol=30,
                        max_harmonic=20,
                        max_fundamentals=3,
                        min_remaining_peaks=5, #how many noise peaks to let through
                        min_score=4):

    remaining_peaks = peak_freqs.copy()
    detected = []

    while len(detected) < max_fundamentals and len(remaining_peaks) > min_remaining_peaks:

        best_f0 = None
        best_score = 0
        best_explained = None

        for f0 in candidate_f0s:

            explained = []

            for p in remaining_peaks:

                n = int(round(p / f0))

                if n < 1 or n > max_harmonic:
                    continue

                error_cents = 1200 * np.log2(p / (n * f0))

                if abs(error_cents) <= cents_tol:
                    explained.append(p)

            score = len(explained)
            ratio = len(explained)/(max(explained)/f0) if len(explained) > 0 else 0
            if score > best_score and ratio > 0.1:
                best_score = score
                best_f0 = f0
                best_explained = explained

        # stop if nothing meaningful found
        if best_score < min_score or best_f0 is None:
            break
        
        detected.append(best_f0)

        # remove explained peaks
        remaining_peaks = remaining_peaks[
            ~np.isin(remaining_peaks, best_explained)
        ]

    return np.array(detected)
def find_fundamentals(peak_freqs):
    means = find_candidates(peak_freqs)
    results = detect_fundamentals(peak_freqs, means)
    return results

# print(find_fundamentals(test_freqs))