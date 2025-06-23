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
EPSILON=1e-8

def get_spectral_args(sample_rate):
     window_size=np.round(sample_rate*WIN_DUR)
     n_overlap=np.round((WIN_DUR*sample_rate)*(1-HOP_FRAC))
     return n_overlap,window_size

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

def handle_SNR_after(f1,origin_sig_psd,anc_sig_psd,ax):
    snr_after = anc_sig_psd / (np.abs(anc_sig_psd - origin_sig_psd) + EPSILON)#X^/N^
    snr_after_dB = 10 * np.log10(snr_after)
    ax.plot(f1, snr_after_dB, label="After ANC", color='orange')
    ax.set_title("SNR After ANC")
    return snr_after

def handle_bp_after(f1,origin_sig_psd,bp_sig_anc_db,fs,fmin,fmax,figure):
        bp_origin_sig_db = 10*np.log10(bandpower(origin_sig_psd, f1, fs, fmin=fmin, fmax=fmax))
        bp_noise_after_db = bp_sig_anc_db - bp_origin_sig_db
        bp_after_db=bp_sig_anc_db-bp_noise_after_db
        figure.text(0.5, 0.00, f"BP {fmin}-{fmax} Hz SNR After ANC: {bp_after_db:.2f} dB", ha='center', fontsize=10)
        return bp_after_db

def handle_bandpowers(f1,full_sig_psd,noise_psd,anc_sig_psd,fs,fmin,fmax,origin_psd,fig):
    bp_sig_before_db=10*np.log10(bandpower(full_sig_psd,f1,fs,fmin=fmin,fmax=fmax))
    bp_noise_db=10*np.log10(bandpower(noise_psd,f1,fs,fmin=fmin,fmax=fmax))
    bp_sig_anc_db=10*np.log10(bandpower(anc_sig_psd,f1,fs,fmin=fmin,fmax=fmax))
    bp_snr_before_db=bp_sig_before_db-bp_noise_db
    bp_delta_n_db=bp_sig_before_db-bp_sig_anc_db
    if origin_psd is not None:
        bp_after_db=handle_bp_after(f1,origin_psd,bp_sig_anc_db,fs,fmin,fmax,fig)
    fig.text(0.5, 0.03, 
    f"BP-{fmin} to {fmax} SNR before ANC: {bp_snr_before_db:.2f} dB",ha='center', fontsize=10)
    fig.text(0.5, 0.01, 
            f"delta noise-{fmin} to {fmax} of(noise before(in the beginning)-noise after ANC): {bp_delta_n_db:.2f} dB",ha='center', fontsize=10)         
    return bp_snr_before_db,bp_delta_n_db,bp_after_db
     
def signal_noise_comparison(full_sig,noise,anc_sig,fs,fmin,fmax,origin_sig=None):
        """SNR comparisons plots-must be time domain signals and then i will do PSD by welch method"""
        NUM_PLOTS=3#just 3 PLOTS, before and after and delta n
        snr_after = None
        origin_sig_psd=None
        fig,ax=plt.subplots(NUM_PLOTS,1)
        f1,full_sig_psd=psd_welch(full_sig,fs)
        f2,noise_psd=psd_welch(noise,fs)
        f3,anc_sig_psd=psd_welch(anc_sig,fs)
       
        if origin_sig is not None:
             f4, origin_sig_psd = psd_welch(origin_sig, fs)
        else:
            origin_sig_psd=None
            f4 = f1  # just for assert check

        # Ensure frequency vectors are consistent-JUST IN CASE
        assert np.allclose(f1, f2) and np.allclose(f1, f3) and np.allclose(f1,f4)
        
        snr_before=full_sig_psd/(noise_psd+EPSILON)
        delta_noise=full_sig_psd/origin_sig_psd

        if origin_sig is not None:
             snr_after=handle_SNR_after(f1,origin_sig_psd,anc_sig_psd,ax[1])

        snr_before_dB=10*np.log10(snr_before)
        delta_noise_db = 10 * np.log10(np.maximum(np.abs(delta_noise), EPSILON))

        ax[0].plot(f1, snr_before_dB, label="Before ANC")
        ax[2].plot(f1,delta_noise_db,label="Change in noise in spectral")
        for a in ax:
            a.set_xlabel("Frequency [Hz]")
            a.set_ylabel("SNR [dB]")
            a.legend()
        ax[0].set_title("SNR Before ANC")
        ax[1].set_title("SNR After ANC")
        ax[2].set_title("delta noise (noise(before anc)-noise(after anc))")
        
        plt.tight_layout()
        plt.show()
        bp_snr_before_db,bp_delta_n_db,bp_after_db=handle_bandpowers(f1,full_sig_psd,noise_psd,anc_sig_psd,fs,fmin,fmax,origin_sig_psd,fig)
        return f1,snr_before_dB,delta_noise_db,bp_snr_before_db,bp_delta_n_db,bp_after_db

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
    return np.log10(np.mean(lin_before)),np.log10(np.std(lin_before+EPSILON))

def bandpower_statics(band_power_before:np.array([float]),band_power_after:np.array([float]),fmin,fmax,signals_name:np.array([str])): # type: ignore
    NUM_PLOTS = 2
    fig, ax = plt.subplots(NUM_PLOTS, 1, figsize=(8, 6), constrained_layout=True)

    # Figure-level title
    fig.suptitle(f"Band Power between {fmin} to {fmax} — Before vs After", fontsize=14)

    # Plot 1
    ax[0].plot(signals_name, band_power_before, marker='o')
    ax[0].set_ylabel('Band Power (dB)')
    ax[0].set_title(f'Before ANC-SNR: {fmin} – {fmax} Hz')
    ax[0].tick_params(axis='x', rotation=45)

    # Plot 2
    ax[1].plot(signals_name, band_power_after, marker='o', color='orange')
    ax[1].set_ylabel('Band Power (dB)')
    ax[1].set_title(f'After ANC-Delta noises: {fmin} – {fmax} Hz')
    ax[1].tick_params(axis='x', rotation=45)
    mean_before,std_before=mean_std(band_power_before)
    mean_after,std_after=mean_std(band_power_after)
    

    fig.text(0.5, 0.03, 
            f"Mean before ANC: {mean_before:.2f} dB | Std: {std_before:.2f} dB", 
            ha='center', fontsize=10)

    fig.text(0.5, 0.01, 
            f"Mean after ANC of noise change (before-after): {mean_after:.2f} dB | Std: {std_after:.2f} dB", 
            ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.show()