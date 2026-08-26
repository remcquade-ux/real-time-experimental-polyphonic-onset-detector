import numpy as np


def _matches(frequency, candidates, cents_tolerance):
    if len(candidates) == 0:
        return False
    cents = 1200 * np.abs(np.log2(frequency / candidates))
    return bool(np.any(cents <= cents_tolerance))


def track_fundamentals(
    results,
    onset_detected,
    state,
    cents_tolerance=50,
    release_frames=3,
):
    """Gate fundamentals so new frequencies require an onset authorization."""
    results = np.asarray(results, dtype=float)
    active = np.asarray(state.get("active", []), dtype=float)
    missing = state.setdefault("missing", {})

    if onset_detected:
        active = results.copy()
        missing = {float(frequency): 0 for frequency in active}
    else:
        matched = []
        next_missing = {}

        for frequency in active:
            match = results[
                1200 * np.abs(np.log2(results / frequency)) <= cents_tolerance
            ] if len(results) else np.array([])

            if len(match):
                matched.append(match[0])
                next_missing[float(match[0])] = 0
            else:
                count = missing.get(float(frequency), 0) + 1
                if count < release_frames:
                    matched.append(frequency)
                    next_missing[float(frequency)] = count

        active = np.asarray(matched, dtype=float)
        missing = next_missing

    state["active"] = active
    state["missing"] = missing

    if len(active):
        values = ", ".join(f"{frequency:.2f} Hz" for frequency in active)
        print(f"Fundamentals: {values}")
    else:
        print("Fundamentals: none")

    return active
