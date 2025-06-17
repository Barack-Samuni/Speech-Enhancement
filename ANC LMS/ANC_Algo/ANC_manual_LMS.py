import numpy as np
import soundfile as sf


def lms_anc(input_file, noise_file, output_file="clean_sig.wav"):
    # Load the signals
    sig_noise, fs1 = sf.read(input_file)  # Primary input: signal + noise
    noise, fs2 = sf.read(noise_file)  # Reference input: noise only

    # Ensure both signals are the same length
    min_len = min(len(sig_noise), len(noise))
    d = sig_noise[:min_len]  # Desired signal
    x = noise[:min_len]  # Reference input signal

    # LMS Parameters
    filter_len = 128
    mu = 0.01
    weights = np.zeros(filter_len)

    # Apply LMS filter
    output = lms_method(filter_len, d, x, mu)

    # Save cleaned signal
    sf.write(output_file, output, fs1)
    print(f"Cleaned signal written to {output_file}")


def lms_method(filter_len, d, x, mu):
    n_samples = len(d)
    weights = np.zeros(filter_len)
    output = np.zeros(n_samples)

    for i in range(filter_len, n_samples):
        x_i = x[i - filter_len:i][::-1]  # Last 'filter_len' samples, reversed
        y = np.dot(weights, x_i)  # Filter output
        e = d[i] - y  # Error signal (desired - output)
        weights += mu * e * x_i  # LMS weight update
        output[i] = e  # Output is the error signal (cleaned signal)

    return output
