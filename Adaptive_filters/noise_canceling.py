from pyroomacoustics.adaptive import NLMS, RLS
import numpy as np
import soundfile as sf
from pathlib import Path
from sig_gen import add_noise_to_audio
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import stft
from AdaptiveFilters import RLSFilter
from SRRLS import SRRLS

import warnings
warnings.simplefilter("error")
## -----------------------------------------------
# helper functions
## -----------------------------------------------
def play_audio(signal, sample_rate):
    """
    Play audio signal using sounddevice.
    
    Parameters:
        signal (np.ndarray): Audio signal to play.
        sample_rate (int): Sample rate of the audio signal.
    """
    sd.play(signal, samplerate=sample_rate)
    sd.wait()  # Wait until the sound has finished playing

def plot_stft(signal, sample_rate, n_fft=1024, hop_length=512, title="STFT Magnitude"):
    """
    Plot the Short-Time Fourier Transform (STFT) magnitude of a signal.

    Parameters:
        signal (np.ndarray): Input audio signal.
        sample_rate (int): Sampling rate of the signal.
        n_fft (int): Number of FFT points.
        hop_length (int): Hop length for STFT.
        title (str): Title for the plot.
    """

    # Compute STFT
    f, t, Zxx = stft(signal, fs=sample_rate, nperseg=n_fft, noverlap=n_fft - hop_length)
    magnitude = np.abs(Zxx)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t, f, 20 * np.log10(magnitude + 1e-10), shading='gouraud', cmap='viridis', norm=Normalize(vmin=-80, vmax=0))
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude [dB]')
    plt.tight_layout()
    plt.show()

## -----------------------------------------------
# PARAMS
## -----------------------------------------------
big_number = 1e6
eps = 1e-9

# speech_path  = Path(r"C:\Users\fireb\Documents\Projects\acoustics\Noise\chepez street walk.wav")
speech_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\signals\p257_023.wav")
# noise_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\Noise\chepez street walk.wav")
noise_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\Noise\screams.wav")
# noise_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\Noise\wind_noise.mp3")
Ourput_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\OUTPUT")
# Add noise to the speech signal
print("Adding noise to the audio...")
noisy_signal, audio, noise, sr = add_noise_to_audio(speech_path, noise_path, noise_factor=2)

# Ensure both have the same length and shape
# Ensure equal length
min_len = min(len(noisy_signal), len(noise))
noise = noise[:min_len]
# noise[noise==0] = eps  # Avoid division by zero in NLMS
# noisy_signal[noisy_signal==0] = eps  # Avoid division by zero in NLMS
noisy_signal = noisy_signal[:min_len]

# add offset
DC = 0
noise += DC
noisy_signal += DC

# plt.figure
# plt.plot(np.arange(0,len(noise))/sr, noise)
# plt.plot(np.arange(0,len(audio))/sr, audio)
# plt.show()
print("Starting noise cancellation...")
# Initialize the NLMS adaptive filter
filter_len = 64
mu = 0.1
n_taps = 64
# rls = RLS
# rls = RLS(64)
lam = 0.999
rls = RLSFilter(n_taps=n_taps, delta=10, lam=lam)


x_data = noisy_signal
d_data = noise
# error matrix and weight matrix 
e = np.ones_like(noisy_signal)
w = np.zeros_like(noisy_signal)
x_buf = np.zeros((filter_len,))  # Buffer for the input signal
N = len(e)
# for i in range(min_len):
#     rls.update( noisy_signal[i], noise[i])

# Main loop: scalar-by-scalar processing
# for i in range(min_len):
#     rls.update( noisy_signal[i], noise[i])
#     x_buf[1:] = x_buf[0:-1]  # nlms(input_sample, desired_output_sample)
#     x_buf[0] = noisy_signal[i]  # Update the buffer with the new input sample
#     e[i] = noisy_signal[i] - np.inner(rls.w, x_buf)  # e = d - y

# rls = RLSFilter(n_taps=10, delta=10)

errors = []
for i in range(N - n_taps + 1):
    x_vec = x_data[i:i+n_taps]
    d = d_data[i]
    y, e = rls.adapt(x_vec, d)
    errors.append(e)

err_array = np.array(errors)

# plot_stft(noisy_signal, sr, title="Noisy Signal STFT")
# plot_stft(noise, sr, title="Error Signal STFT")


# Error is the cleaned signal
print("Cleaning the signal...")
min_len = np.min([len(noisy_signal), len(err_array)])
cleaned_signal = err_array[:min_len]
plot_stft(err_array, sr, title="Denoised")
plot_stft(noisy_signal, sr, title="original")
# plot_stft(cleaned_signal, sr, title="Cleaned Signal STFT")
# plot_stft(noisy_signal, sr)
# Save result
print("Saving the denoised output...")
if not Ourput_path.exists():
    Ourput_path.mkdir(parents=True)
sf.write(Ourput_path / "denoised_output_rls_screems.wav", 3*cleaned_signal, sr)
sf.write(Ourput_path / "Original_noised_screems.wav", noisy_signal, sr)
