import soundfile as sf
from pathlib import Path
from Filters import FilterWrapper
from AdaptiveFilters import RLSFilter
from sig_gen import add_noise_to_audio, plot_stft

Ourput_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\OUTPUT")

speech_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\signals\races.wav")
noise_path = Path(r"C:\Users\fireb\Documents\Projects\acoustics\Noise\wind_noise.mp3")

noisy_signal, audio, noise, sr = add_noise_to_audio(speech_path, noise_path, noise_factor=3)

n_taps = 64
filt = FilterWrapper(RLSFilter, n_taps=n_taps)

y, e = filt.process(noisy_signal=noisy_signal, noise=noise)

# plot_stft(e, sr, title="denoised")

sf.write(Ourput_path / "denoised_output_rls_wind_omer.wav", e, sr)
sf.write(Ourput_path / "Original_noised_wind_omer.wav", noisy_signal, sr)
sf.write(Ourput_path / "distorted_wind_noise.wav", noise, sr)


