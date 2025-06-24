import soundfile as sf
import numpy as np
from pathlib import Path
import inspect


def add_noise_to_audio(audio_path, noise_path, output_path=None, noise_factor=1.0):
    """
    Loads audio and noise files, adds noise to the audio, and optionally saves the result.

    Parameters:
        audio_path (str or Path): Path to the clean audio file.
        noise_path (str or Path): Path to the noise audio file.
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

    # Normalize to prevent clipping
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 1.0:
        noisy_audio = noisy_audio / max_val

    if output_path:
        sf.write(output_path, noisy_audio, sr_audio)

    return noisy_audio, audio, noise, sr_audio
