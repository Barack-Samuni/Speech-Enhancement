from pyroomacoustics.adaptive import NLMS
import numpy as np
import soundfile as sf

# Load input signals
total_sig, sr = sf.read(r"C:\Users\Omer\Documents\Projects\WAVS\ANC WAVS\total_sig.wav")
noise, _  = sf.read(r"C:\Users\Omer\Documents\Projects\WAVS\ANC WAVS\processed_noise.wav")

# Use only first channel if stereo
if total_sig.ndim > 1:
    total_sig = total_sig.mean(axis=1)
if noise.ndim > 1:
    noise = noise.mean(axis=1)

# Match lengths
min_len = min(len(total_sig), len(noise))
total_sig = total_sig[:min_len]
noise = noise[:min_len]

# Init NLMS filter
filter_len = 128
l = NLMS(filter_len, mu=0.01)
e = np.zeros(min_len)

# Adaptive filtering loop
for n in range(min_len):
    l.update(noise[n], total_sig[n])
    y = np.dot(l.w, l.x)
    e[n] = total_sig[n] - y  # cleaned signal

# Save output
sf.write(r"C:\Users\Omer\Documents\Projects\WAVS\ANC WAVS\denoised_output.wav", e, sr)
print("yess")
