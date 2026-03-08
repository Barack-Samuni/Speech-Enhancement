import soundfile as sf
from pathlib import Path
from Filters import FilterWrapper
from AdaptiveFilters import RLSFilter
from sig_gen import add_noise_to_audio, plot_stft
from NoiseReduction import rnnoise_denoise
from Gain import adaptive_gain


Ourput_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\OUTPUT")

# speech_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\signals\races.wav")
# noise_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\Noise\wind_noise.mp3")
noisy_sig_path = Path(r"C:\Users\fireb\Documents\Graphs and Stuff\signal_tests\1.wav")

sig, fs = sf.read(noisy_sig_path)
sig_gain = adaptive_gain(sig[fs*20:fs*30], fs)
sig_eh, fs_new = rnnoise_denoise(sig_gain, fs)
plot_stft(sig[fs*20:fs*30],fs, title="before")
plot_stft(sig_gain,fs, title="before_gain")
plot_stft(sig_eh,fs_new, title="after_w/GAIN")
# sf.write(Ourput_path / "1_rnnoised_gain_part2.wav", sig_eh, fs_new)


# noisy_signal, audio, noise, sr = add_noise_to_audio(speech_path, noise_path, noise_factor=3)

# n_taps = 64
# filt = FilterWrapper(RLSFilter, n_taps=n_taps)

# y, e = filt.process(noisy_signal=noisy_signal, noise=noise)

# plot_stft(e, sr, title="denoised")

# sf.write(Ourput_path / "denoised_output_rls_wind_omer.wav", e, sr)
# sf.write(Ourput_path / "Original_noised_wind_omer.wav", noisy_signal, sr)
# sf.write(Ourput_path / "distorted_wind_noise.wav", noise, sr)


