import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.signal as sp
from scipy.ndimage import uniform_filter
from scipy.integrate import trapz
import matplotlib.pyplot as plt
# spectral argsp
WIN_DUR = 0.064
HOP_FRAC = 0.2

def get_spectral_args(sample_rate):
     window_size=np.round(sample_rate*WIN_DUR)
     n_overlap=np.round((WIN_DUR*sample_rate)*(1-HOP_FRAC))

def create_spectogram(sig,fs):
    n_overlap,window_size=get_spectral_args(fs)
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
    assert np.allclose(f1, f2), "Frequency vectors do not match"
    assert np.allclose(f2, f3), "Frequency vectors do not match" 
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



# def smooth_data(fs,sig,f):
#     """To smooth the signals just for the comparison plot"""
#     window_size=np.round(fs*WIN_DUR)
#     sig=uniform_filter(sig,window_size,mode="reflect")
#     f=uniform_filter(f,window_size,mode="reflect")
#     return f,sig
#      #'reflect' mirrors your signal across the border.


def psd_welch(sig,fs,nfft:int=8192):
    n_overlap,window_size=get_spectral_args(fs)
    f,psd_sig=sp.welch(sig,fs,nperseg=window_size,noverlap=n_overlap,nfft=nfft)
    return f, psd_sig

     
def signal_noise_comparison(sig,noise,anc_sig,fs,freq_vector,fmin,fmax):
        """SNR comparisons plots-must be time domain signals and then i will do PSD by welch method"""
        NUM_PLOTS=2#just 2 PLOTS, before and after
        EPSILON=1e-8
        fig,ax=plt.subplots(NUM_PLOTS,1)
        f1,sig_psd=psd_welch(fs,sig)
        f2,noise_psd=psd_welch(fs,noise)
        f3,anc_sig_psd=psd_welch(fs,anc_sig)
        #f1,2,3 must be equal because of len(noise)==len(sig) and fs(sig)=fs(noise)

        # Ensure frequency vectors are consistent-JUST IN CASE
        assert np.allclose(f1, f2), "Frequency vectors do not match"
        assert np.allclose(f2, f3), "Frequency vectors do not match"        
        
        snr_before=sig_psd/(noise_psd+EPSILON)
        snr_after=anc_sig_psd/(noise_psd+EPSILON)
        snr_before_dB=10*np.log10(snr_before)
        snr_after_dB=10*np.log10(snr_after)
        ax[0].plot(f1, snr_before_dB, label="Before ANC")
        ax[1].plot(f1, snr_after_dB, label="After ANC", color='orange')
        for a in ax:
            a.set_xlabel("Frequency [Hz]")
            a.set_ylabel("SNR [dB]")
            a.legend()
        ax[0].set_title("SNR Before ANC")
        ax[1].set_title("SNR After ANC")
        plt.show()

        bp_snr_before=10*np.log10(bandpower(snr_before,fs,fmin,fmax))
        bp_snr_after=10*np.log10(bandpower(snr_after,fs,fmin,fmax))
        return f1,snr_before_dB,snr_after_dB,bp_snr_before,bp_snr_after

def bandpower(pxx,freqs, fs, fmin:float=300, fmax:float=5000):
    """
    Compute band power in [fmin, fmax] using ready equalized psd and freqs
    """
    assert 0 <= fmin < fmax <= fs/2, "Frequency limits must satisfy 0 ≤ fmin < fmax ≤ fs/2"

    idx = (freqs >= fmin) & (freqs <= fmax)
    return trapz(pxx[idx], freqs[idx])

def mean_std(band_power:np.array):
    """gets band powers array in decibels and brings back expectation and std of it"""
    lin_before=np.power(10,band_power/10)
    return np.log10(np.mean(lin_before)),np.log10(np.std(lin_before))

def bandpower_statics(band_power_before:np.array([float]),band_power_after:np.array([float]),fmin,fmax,signals_name:np.array([str])):
    NUM_PLOTS = 2
    fig, ax = plt.subplots(NUM_PLOTS, 1, figsize=(8, 6), constrained_layout=True)

    # Figure-level title
    fig.suptitle(f"Band Power between {fmin} to {fmax} — Before vs After", fontsize=14)

    # Plot 1
    ax[0].plot(signals_name, band_power_before, marker='o')
    ax[0].set_ylabel('Band Power (dB)')
    ax[0].set_title(f'Before ANC: {fmin} – {fmax} Hz')
    ax[0].tick_params(axis='x', rotation=45)

    # Plot 2
    ax[1].plot(signals_name, band_power_after, marker='o', color='orange')
    ax[1].set_ylabel('Band Power (dB)')
    ax[1].set_title(f'After ANC: {fmin} – {fmax} Hz')
    ax[1].tick_params(axis='x', rotation=45)
    mean_before,std_before=mean_std(band_power_before)
    mean_after,std_after=mean_std(band_power_after)

    fig.text(0.5, 0.03, 
            f"Mean before ANC: {mean_before:.2f} dB | Std: {std_before:.2f} dB", 
            ha='center', fontsize=10)

    fig.text(0.5, 0.01, 
            f"Mean after ANC: {mean_after:.2f} dB | Std: {std_after:.2f} dB", 
            ha='center', fontsize=10)
    
    plt.show()