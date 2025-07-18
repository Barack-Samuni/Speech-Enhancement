import soundfile as sf
import numpy as np
import sounddevice as sd
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.signal import lfilter, butter
from scipy.signal import stft

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
    plt.pcolormesh(t, f, 20 * np.log10(magnitude + 1e-10), shading='gouraud', cmap='inferno', norm=Normalize(vmin=-80, vmax=0))
    plt.title(title)
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.colorbar(label='Magnitude [dB]')
    plt.tight_layout()
    plt.show()


def play_audio(signal, sample_rate):
    """
    Play audio signal using sounddevice.
    
    Parameters:
        signal (np.ndarray): Audio signal to play.
        sample_rate (int): Sample rate of the audio signal.
    """
    sd.play(signal, samplerate=sample_rate)
    sd.wait()  # Wait until the sound has finished playing

def add_noise_to_audio(audio_path, noise_path, distortion_flag = True, output_path=None, noise_factor=1.0):
    """
    Loads audio and noise files, adds noise to the audio, and optionally saves the result.

    Parameters:
        audio_path (str or Path): Path to the clean audio file.
        noise_path (str or Path): Path to the noise audio file.
        distortion_flag (logical, optional): by default adds distortion to noise to make it more challanging to filter
        output_path (str or Path, optional): If provided, saves the noisy audio to this path.
        noise_factor (float, optional): Factor to control the amount of noise added. Default is 1.0.

    Returns:
        np.ndarray: The noisy audio samples.
        int: The sample rate.
    """
    audio, sr_audio = sf.read(audio_path)
    noise, sr_noise = sf.read(noise_path)

    # Ensure both signals are mono or match channels
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    # Resample noise if sample rates differ
    if sr_audio != sr_noise:
        # Interpolate noise to match audio's sample rate
        duration_noise = len(noise) / sr_noise
        num_samples = int(duration_noise * sr_audio)
        noise = np.interp(
            np.linspace(0, len(noise), num_samples, endpoint=False),
            np.arange(len(noise)),
            noise
        )
        sr_noise = sr_audio

    # Match noise length to audio length
    if len(noise) < len(audio):
        repeats = int(np.ceil(len(audio) / len(noise)))
        noise = np.tile(noise, repeats)
    noise = noise[:len(audio)]

    noisy_audio = audio + noise_factor * noise

    if distortion_flag:
        # FIR room impulse response (mild echo tail)
        room_ir = np.exp(-np.linspace(0, 1, 20)) * np.random.choice([1, -1], 20)
        noise = lfilter(room_ir, [1], noise)

        # IIR mic coloration: lowpass + slight resonance
        # b, a = butter(2, 0.3, btype='low')  # simple lowpass
        # noise = lfilter(b, a, noise_reverb)
        noise = noise / np.max(np.abs(noise))

    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val

    if output_path:
        sf.write(output_path, noisy_audio, sr_audio)

    return noisy_audio, audio, noise, sr_audio
