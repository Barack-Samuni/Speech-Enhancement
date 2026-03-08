import numpy as np
import scipy.signal as signal

# -------------------------
# Adaptive gain control
# -------------------------

def adaptive_gain(
    y,
    sr,
    max_boost_db=12.0
):
    """
    Adaptive gain normalization based on local maxima.
    
    Parameters
    ----------
    y : np.ndarray
        Mono audio signal (float32, -1..1)
    sr : int
        Sample rate
    max_boost_db : float
        Safety limit on gain boost

    Returns
    -------
    y_out : np.ndarray
        Gain-adjusted signal
    """

    eps = 1e-8

    # 1. Envelope extraction (speech-friendly)
    # Hilbert gives smooth amplitude without rectification artifacts
    env = np.abs(signal.hilbert(y))

    # Smooth envelope (~40 ms window)
    win_len = int(0.04 * sr)
    if win_len % 2 == 0:
        win_len += 1
    env_smooth = signal.medfilt(env, kernel_size=win_len)

    # 2. Detect local maxima in envelope
    # Min distance ≈ 80 ms (speech syllable scale)
    min_distance = int(0.08 * sr)
    peaks, _ = signal.find_peaks(env_smooth, distance=min_distance)

    if len(peaks) == 0:
        return y  # nothing to do

    peak_vals = env_smooth[peaks]

    # 3. Global reference (robust, not absolute max)
    global_ref = np.percentile(peak_vals, 95) + eps

    # 4. Compute gain at peaks
    gain_peaks = global_ref / (peak_vals + eps)

    # Limit excessive boosting
    max_boost = 10 ** (max_boost_db / 20)
    gain_peaks = np.clip(gain_peaks, 1.0, max_boost)

    # 5. Interpolate gain envelope over time
    gain_env = np.ones_like(y)
    gain_env[peaks] = gain_peaks

    # Linear interpolation between peaks
    gain_env = np.interp(
        np.arange(len(y)),
        peaks,
        gain_env[peaks]
    )

    # 6. Temporal smoothing of gain (avoid pumping)
    # Attack ~100 ms, release ~300 ms
    attack = int(0.10 * sr)
    release = int(0.30 * sr)

    gain_env = smooth_gain(gain_env, attack, release)

    # 7. Apply gain safely
    y_out = y * gain_env
    y_out = np.clip(y_out, -1.0, 1.0)

    return y_out


def smooth_gain(gain, sr, attack_time=0.10, release_time=0.30):
    """
    Exponential smoothing of gain envelope with proper time constants.

    Parameters
    ----------
    gain : np.ndarray
        Raw gain envelope
    sr : int
        Sample rate
    attack_time : float
        Attack time in seconds
    release_time : float
        Release time in seconds
    """

    eps = 1e-8

    alpha_attack = np.exp(-1.0 / (attack_time * sr + eps))
    alpha_release = np.exp(-1.0 / (release_time * sr + eps))

    smoothed = np.zeros_like(gain)
    smoothed[0] = gain[0]

    for i in range(1, len(gain)):
        if gain[i] > smoothed[i - 1]:
            alpha = alpha_attack
        else:
            alpha = alpha_release

        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * gain[i]

    return smoothed

