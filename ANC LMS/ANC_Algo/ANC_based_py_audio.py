from pyroomacoustics.adaptive import lms,NLMS,SubbandLMS,AdaptiveFilter
import numpy as np
import soundfile as sf

# Load your audio (signal + noise) and reference (noise only)
d, sr = sf.read("primary_signal.wav")    # signal + noise
x, _  = sf.read("reference_noise.wav")   # noise only

# Ensure both have the same length and shape
min_len = min(len(d), len(x))
d = d[:min_len]
x = x[:min_len]

# Initialize LMS
filter_len = 128
l = lms(filter_len, mu=0.01)
y = np.zeros_like(d)

# Apply adaptive filtering
for i in range(min_len):
    y[i] = l.adapt(x[i], d[i])

# Error is the cleaned signal
cleaned_signal = d - y

# Save result
sf.write("denoised_output.wav", cleaned_signal, sr)