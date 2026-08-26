import numpy as np
import matplotlib.pyplot as plt

def save_three_period_frame(wave, sample_rate, out_dir):
    """
    Saves an image of the first 3 periods of a waveform frame.

    Parameters
    ----------
    wave : np.ndarray
        Time-domain waveform (single frame)
    sample_rate : int
        Audio sample rate
    frame_index : int
        Used in filename
    out_dir : str
        Output folder
    """

    # remove DC
    wave = wave - np.mean(wave)

    # autocorrelation
    corr = np.correlate(wave, wave, mode='full')
    corr = corr[len(corr)//2:]

    # ignore zero-lag peak
    min_period = int(sample_rate / 2000)  # avoid tiny noise peaks
    max_period = int(sample_rate / 40)    # ~40Hz lower bound

    period = np.argmax(corr[min_period:max_period]) + min_period

    # crop first 3 periods
    end = min(len(wave), period * 3)
    segment = wave[:end]

    # normalize
    max_val = np.max(np.abs(segment))
    if max_val > 0:
        segment = segment / max_val

    # plot
    plt.figure(figsize=(6,3))
    plt.plot(segment)
    plt.ylim(-1.1, 1.1)
    plt.axis('off')

    # save
    plt.savefig(f"{out_dir}/frame_{00000:05d}.png",
                bbox_inches='tight',
                pad_inches=0)
    plt.close()