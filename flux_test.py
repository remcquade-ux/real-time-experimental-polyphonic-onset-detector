import numpy as np
import sounddevice as sd

# Your spectral flux class from before
class SpectralFlux:
    def __init__(self, sr):
        self.sr = sr
        self.hop = int(sr * 0.001)       # 5 ms hop
        self.fft_size = 2*self.hop     # 20 ms window
        self.window = np.hanning(self.fft_size)
        self.prev_mag = None
        self.max_flux = 1e-12

    def process_frame(self, frame):
        frame = frame * self.window
        spectrum = np.fft.rfft(frame)
        mag = np.abs(spectrum)
        mag = np.log(mag + 1e-12)

        if self.prev_mag is None:
            self.prev_mag = mag
            return 0.0

        diff = mag - self.prev_mag
        diff[diff < 0] = 0
        flux = np.sum(diff)
        self.prev_mag = mag

        self.max_flux = max(self.max_flux, flux)
        norm_flux = flux / self.max_flux

        return norm_flux

# Real-time callback
def audio_callback(indata, frames, time, status):
    global buffer
    buffer = np.concatenate((buffer, indata[:,0]))  # mono
    while len(buffer) >= detector.fft_size:
        frame = buffer[:detector.fft_size]
        flux = detector.process_frame(frame)
        if flux > 0.6:  # example threshold
            print(f"Normalized flux: {flux:.3f}")
        buffer = buffer[detector.hop:]  # advance by hop

# Setup
sr = 44100
detector = SpectralFlux(sr)
buffer = np.array([], dtype=np.float32)

# Start stream
with sd.InputStream(channels=1, samplerate=sr, callback=audio_callback, blocksize=int(sr*0.001)):
    print("Streaming... press Ctrl+C to stop")
    import time
    while True:
        time.sleep(0.001)