from pyroomacoustics.adaptive import NLMS
import numpy as np
import soundfile as sf
import scipy.signal as sp
import matplotlib.pyplot as plt

# spectral argsp
WIN_DUR = 0.064
HOP_FRAC = 0.2


def create_spectogram(sig,fs):
    window_size=np.round(fs*WIN_DUR)
    n_overlap=np.round((WIN_DUR*fs)*(1-HOP_FRAC))
    f1,t1,sig_stft=sp.stft(x=sig,fs=fs,noverlap=n_overlap,nperseg=window_size)
    psd_sig=np.abs(sig_stft)**2
    return f1,t1,psd_sig


def paint_spectogram(psd,ax):
    ax.imshow(10*np.log10(psd),origin='lower',aspect='auto',cmap='inferno',vmin=-90,vmax=-20)

def plot_spectograms_of_all(total_sig,noise,cleaned_sig,fs1,kmin=0,kmax=200):
    f1,t1,psd_total=create_spectogram(total_sig,fs1)
    f2,t2,psd_noise=create_spectogram(noise,fs1)
    f3,t3,psd_clean=create_spectogram(cleaned_sig,fs1)
    dict_psds = {
    "Before ANC": psd_total,
    "Only Noise": psd_noise,
    "Cleaned ANC Noise": psd_clean
}
    psds_titles=list(dict_psds.keys())
    psds=list(dict_psds.values())
    length_psds=3
    if length_psds == 1:
        ax = [ax]
    fig,ax=plt.subplots(length_psds,1)
    for i in range(length_psds):
        ax[i].set_ylim([kmin,kmax])
        paint_spectogram(psds[i],ax[i])
        ax[i].set_title(psds_titles[i])
    plt.tight_layout()
    plt.show()

def NLMS_calculation(total_sig,noise,fs1,fs2):

    assert fs1 == fs2, f"Sampling rates do not match: fs1={fs1}, fs2={fs2}"

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
    sf.write(r"C:\Users\Omer\Documents\Projects\WAVS\ANC WAVS\הרצת ניסויי\Anc_cleaned_output.wav", e, fs1)
    print("Anc Completed and saved!")
    return e



def main():
        # Load input signals
    total_sig, fs1 = sf.read(r"C:\Users\Omer\Documents\Projects\WAVS\ANC WAVS\הרצת ניסויי\total_sig.wav")
    noise, fs2  = sf.read(r"C:\Users\Omer\Documents\Projects\WAVS\ANC WAVS\הרצת ניסויי\processed_noise.wav")
    cleaned_sig=NLMS_calculation(total_sig,noise,fs1,fs2)
    plot_spectograms_of_all(total_sig,noise,cleaned_sig,fs1)


if __name__=="__main__":
    main()