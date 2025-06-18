from pyroomacoustics.adaptive import NLMS
import numpy as np
import soundfile as sf
import scipy.signal as sp
import matplotlib.pyplot as plt

def resample_fs(sig,fs_old,fs_new):
    num_samples = int(len(sig) * fs_new / fs_old)
    resampled_sig = sp.resample(sig, num_samples)
    return resampled_sig,fs_new

# spectral argsp
WIN_DUR = 0.064
HOP_FRAC = 0.2


def create_spectogram(sig,fs):
    window_size=np.round(fs*WIN_DUR)
    n_overlap=np.round((WIN_DUR*fs)*(1-HOP_FRAC))
    f1,t1,sig_stft=sp.stft(x=sig,fs=fs,noverlap=n_overlap,nperseg=window_size)
    psd_sig=np.abs(sig_stft)**2
    return f1,t1,psd_sig


def paint_spectogram(psd,ax,t,f):
    ax.imshow(10*np.log10(psd),origin='lower',aspect='auto',cmap='inferno',vmin=-90,vmax=-20,)
# extent=[t[0], t[-1], f[0], f[-1]]
def plot_spectograms_of_all(total_sig,noise,cleaned_sig,fs1,kmin=0,kmax=200):
    """ create Specotgram to all 3 signals based on the signals, sample rate"""
    f1,t1,psd_total=create_spectogram(total_sig,fs1)
    f2,t2,psd_noise=create_spectogram(noise,fs1)
    f3,t3,psd_clean=create_spectogram(cleaned_sig,fs1)
    freqs=[f1,f2,f3]
    times=[t1,t2,t3]
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
        paint_spectogram(psds[i],ax[i], times[i], freqs[i])
        ax[i].set_title(psds_titles[i])
    plt.tight_layout()
    plt.show()

def NLMS_calculation(total_sig,noise,fs1,fs2):
    """ NLMS Calc"""
    if fs1>fs2:
        sig,fs1=resample_fs(sig,fs_old=fs1,fs_new=fs2)

    elif fs1<fs2:
        noise,fs2=resample_fs(sig,fs_old=fs2,fs_new=fs1)

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

    # Adaptive filtering loop- update weights for each sample
    for n in range(min_len):
        l.update(noise[n], total_sig[n])
        y = np.dot(l.w, l.x)
        e[n] = total_sig[n] - y  # cleaned signal

    # Save output
    sf.write(r"C:\Users\galon\Documents\projects\Wavs\exp1\Anc_cleaned_output1.wav", e, fs1)
    print("Anc Completed and saved!")
    return e



def main():
        # Load input signals
    total_sig, fs1 = sf.read(r"C:\Users\galon\Documents\projects\Wavs\exp1\full_sig_noise_sig2.wav")
    noise, fs2  = sf.read(r"C:\Users\galon\Documents\projects\Wavs\exp1\exp_noise2.wav")
    cleaned_sig=NLMS_calculation(total_sig,noise,fs1,fs2)
    plot_spectograms_of_all(total_sig,noise,cleaned_sig,fs1)


if __name__=="__main__":
    main()