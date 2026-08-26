import numpy as np
import sounddevice as sd
import matplotlib.pyplot as plt
import collections

from skew_test import process_frame
from wave_test import save_three_period_frame
from wave_reconstruction import reconstruct_wave_rfft
from noise_filter import noise_filter
from peak_filter import peak_filter
from find_fundamentals import find_fundamentals
from spectral_comb import spectral_comb
from spectral_flux import detect_note_change
from fundamental_tracker import track_fundamentals
# ==============================
# SETTINGS
# ==============================

SAMPLE_RATE = 44100
FRAME_MS = 200
FRAME_SIZE = int(SAMPLE_RATE * FRAME_MS / 1000)##num samples

DB_MIN = -90
DB_MAX = 10

FMIN = 65.41
MAX_FREQ = 10000
DELTA_HZ = MAX_FREQ/int(SAMPLE_RATE * FRAME_MS)


class RealTimeFFT:

    def __init__(self):
        self.audio_buffer = collections.deque(maxlen=FRAME_SIZE * 3)

        self.history = collections.deque(maxlen=1)
        self.fundamental_state = {}
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.line, = self.ax.plot([], [])

        self.ax.set_xlim(FMIN, MAX_FREQ)
        self.ax.set_ylim(DB_MIN, DB_MAX)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Amplitude (dB)")
        self.ax.set_title("Real-Time FFT Spectrum")
        self.ax.grid(True)

        self.valid_scatter = self.ax.scatter([], [], c='lime', s=45, marker='o', zorder=3)
        plt.show(block=False)

    def audio_callback(self, indata, frames, time, status):
        if status:
            print(status)
        self.audio_buffer.extend(indata[:, 0])

    def get_frame(self):
        if len(self.audio_buffer) < FRAME_SIZE:
            return None
        frame = np.array(list(self.audio_buffer)[-FRAME_SIZE:])
        return frame

    def compute_fft(self, frame):
        """
        Standard linear-frequency FFT
        """

        # Apply window to reduce leakage
        window = np.hanning(len(frame))
        y = frame * window

        # FFT
        fft = np.fft.rfft(y)
        mags = np.abs(fft)
        phase = np.angle(fft)

        # Convert to dB
        spectrum_db = 20 * np.log10(mags + 1e-12)
        spectrum_db = np.clip(spectrum_db, DB_MIN, DB_MAX)

        # Frequency axis (Hz)
        freqs = np.fft.rfftfreq(len(y), 1 / SAMPLE_RATE)

        return freqs, spectrum_db, mags, phase

    def run(self):
        print("Running real-time FFT...")

        with sd.InputStream(callback=self.audio_callback,
                            channels=1,
                            samplerate=SAMPLE_RATE):
            while True:
                frame = self.get_frame()
                if frame is None:
                    continue

                # Linear frequency FFT
                freqs, spectrum_db, spectrum, phase = self.compute_fft(frame)

                # Apply noise filter
                filtered, peak_freqs = noise_filter(spectrum_db, freqs)
                combed_peaks = spectral_comb(filtered)


                # Reconstruct wave
                process_frame(10**(filtered/20), phase, SAMPLE_RATE)
                ##use combed to do another find_fundamentals with sorting strat
                mask = (freqs >= FMIN) & (freqs <= MAX_FREQ)
                results = find_fundamentals(peak_freqs)
                onset_detected = detect_note_change(
                    spectrum,
                    self.history[-1]["peaks"] if self.history else spectrum,
                    threshold=5,
                )
                results = track_fundamentals(
                    results,
                    onset_detected,
                    self.fundamental_state,
                )

                fundamental_spectrum = np.full_like(spectrum_db, DB_MIN)
                if len(results):
                    fundamental_indices = np.abs(
                        freqs[:, None] - results
                    ).argmin(axis=0)
                    fundamental_spectrum[fundamental_indices] = spectrum_db[
                        fundamental_indices
                    ]

                self.line.set_data(freqs[mask], fundamental_spectrum[mask])
                
                self.history.append({
                "freqs": freqs,
                "peaks": spectrum
                })

                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause((FRAME_MS / 1000)/10)


if __name__ == "__main__":
    app = RealTimeFFT()
    app.run()