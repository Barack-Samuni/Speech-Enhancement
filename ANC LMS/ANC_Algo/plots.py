import numpy as np
import soundfile as sf
from pathlib import Path
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


def paint_spectogram(psd,ax,t,f):
    im=ax.imshow(10*np.log10(psd),origin='lower',aspect='auto',cmap='inferno',vmin=-90,vmax=-20)
    cbar=ax.figure.colorbar(im,ax=ax)
    cbar.set_label("Power[dB]", rotation=270, labelpad=15)
    
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

