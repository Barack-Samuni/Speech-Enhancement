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
     """spectral args function to stft and WELCH"""
     window_size=np.round(sample_rate*WIN_DUR)
     n_overlap=np.round((WIN_DUR*sample_rate)*(1-HOP_FRAC))
     return n_overlap,window_size

def create_spectogram(sig,fs,nfft:int=8192):
    """STFT FUNCTION BASED ON SCIPY"""
    n_overlap,window_size=get_spectral_args(fs)
    f1,t1,sig_stft=sp.stft(x=sig,fs=fs,noverlap=n_overlap,nperseg=window_size,nfft=nfft)
    psd_sig=np.abs(sig_stft)**2
    return f1,t1,psd_sig


def paint_spectogram(psd,ax,t,f):
    """PRESTING GENERAL SPECTOGRAM"""
    im=ax.imshow(10*np.log10(psd),origin='lower',aspect='auto',cmap='inferno',vmin=-90,vmax=-20)
    cbar=ax.figure.colorbar(im,ax=ax)
    # ax.set_xticks(np.linspace(t[0], t[-1], num=20))
    # ax.set_yticks(np.linspace(f[0], f[-1], num=5000))
    cbar.set_label("Power[dB]", rotation=270, labelpad=15)
    # ax.set_xlabel("Time [s]")
    # ax.set_ylabel("Frequency [Hz]")
    #extent=[t[0], t[-1], f[0], f[-1]]

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



def smooth_data(sig,f):
    """To smooth the signals just for the comparison plot-Uniform filter of SCIPY"""
    window_size=20
    sig=uniform_filter(sig,window_size,mode="reflect")
    f=uniform_filter(f,window_size,mode="reflect")
    return f,sig
     #'reflect' mirrors your signal across the border.


def psd_welch(sig,fs,nfft:int=8192):
    """Welch method to get PSD of the signals by Certain Spectral Args"""
    n_overlap,window_size=get_spectral_args(fs)
    f,psd_sig=sp.welch(sig,fs,nperseg=window_size,noverlap=n_overlap,nfft=nfft)
    return f, psd_sig

# def handle_SNR_after(f1,origin_sig_psd,anc_sig_psd,ax):
#     snr_after = anc_sig_psd / (np.abs(anc_sig_psd - origin_sig_psd) + EPSILON)#X^/N^
#     snr_after_dB = 10 * np.log10(snr_after)
#     ax.plot(f1, snr_after_dB, label="After ANC", color='pink')
#     ax.set_title("SNR After ANC")
#     return snr_after

# def handle_bp_after(f1,origin_sig_psd,bp_sig_anc,fs,fmin,fmax,figure):
#         bp_origin_sig =(bandpower(origin_sig_psd, f1, fs, fmin=fmin, fmax=fmax))
#         bp_estimated_noise=np.abs(bp_sig_anc-bp_origin_sig)+EPSILON
#         snr_after_bp_db = 10*np.log10(bp_origin_sig/bp_estimated_noise)
#         figure.text(0.5, 0.00, f"BP {fmin}-{fmax} Hz SNR After ANC: {snr_after_bp_db:.2f} dB", ha='center', fontsize=10)
#         return snr_after_bp_db


def handle_new_snr(fs,f,full_sig_psd,anc_sig_psd,snr_before_psd):
     """
     based on PARSAVL because it is energy we can divide in frequency domain
     Final equation of our new Spectral SNR=(S*snr)/(anc_sig*(snr+1)-S)
     """
     numerator=(full_sig_psd*snr_before_psd)
     denumerator=np.abs(anc_sig_psd*(snr_before_psd+1)-full_sig_psd)
     numerator_bp=bandpower(numerator,f,fs)
     denumerator_bp=bandpower(denumerator,f,fs)
     snr_after_psd=numerator/(denumerator)
     snr_after_bp=numerator_bp/denumerator_bp
     return snr_after_psd,snr_after_bp
     
def handle_old_snr(fs,f,full_sig_psd,noise_sig_psd):
     """
     Full Signal divided By noise
     """
     numerator_bp=bandpower(full_sig_psd,f,fs)
     denumerator_bp=bandpower(noise_sig_psd,f,fs)
     snr_before_psd=full_sig_psd/noise_sig_psd
     snr_before_bp=numerator_bp/denumerator_bp
     return snr_before_psd,snr_before_bp
     

     

def handle_bandpowers(before_snr_bp,after_snr_bp,fig:plt.figure,fmin:float=300,fmax:float=5000):
    """presents The bandpower on the graphs"""
    bp_snr_before_db=10*np.log10(before_snr_bp)
    bp_after_bp_db=10*np.log10(after_snr_bp)
    fig.text(0.5, 0.03, 
    f"BP-{fmin} to {fmax} SNR before ANC: {bp_snr_before_db:.2f} dB",ha='center', fontsize=10)
    fig.text(0.5, 0.01, 
            f"BP-{fmin} to {fmax} SNR after ANC : {bp_after_bp_db:.2f} dB",ha='center', fontsize=10)         
    return bp_snr_before_db,bp_after_bp_db


          
def signal_noise_comparison(full_sig,noise,anc_sig,fs,fmin:float=300,fmax:float=5000):
        """SNR comparisons plots-must be time domain signals and then i will do PSD by welch method"""
        NUM_PLOTS=2#just 2
        fig,ax=plt.subplots(NUM_PLOTS,1)
        f1,full_sig_psd=psd_welch(full_sig,fs)
        f2,noise_psd=psd_welch(noise,fs)
        f3,anc_sig_psd=psd_welch(anc_sig,fs)
       
        # Ensure frequency vectors are consistent-JUST IN CASE
        assert np.allclose(f1, f2) and np.allclose(f1, f3) 
        
        snr_before_psd,snr_before_bp=handle_old_snr(fs,f1,full_sig_psd,noise_psd)

        snr_after_psd,snr_after_bp=handle_new_snr(fs,f1,full_sig_psd,anc_sig_psd,snr_before_psd)
        _,snr_before_psd=smooth_data(snr_before_psd,f1)#smoothing the SNR signals by uniform filter
        f1,snr_after_psd=smooth_data(snr_after_psd,f1)
        snr_before_dB_psd=10*np.log10(snr_before_psd)
        snr_after_dB_psd=10*np.log10(snr_after_psd)

        ax[0].plot(f1, snr_before_dB_psd, label="Before ANC",color="blue")
        ax[1].plot(f1,snr_after_dB_psd,label="Change in noise in spectral",color="orange")
        for a in ax:
            a.set_xlabel("Frequency [Hz]")
            a.set_ylabel("SNR [dB]")
            a.legend()
        ax[0].set_title("SNR Before ANC")
        ax[1].set_title("SNR After ANC")
        
        plt.tight_layout()
   

        bp_snr_before_db,bp_snr_after_db=handle_bandpowers(snr_before_bp,snr_after_bp,fig,fmin,fmax)
        plt.show()
        return bp_snr_before_db,bp_snr_after_db

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
    return 10*np.log10(np.mean(lin_before)),10*np.log10(np.std(lin_before+EPSILON))

def bandpower_statics(band_power_before:np.array([float]),band_power_after:np.array([float]),fmin,fmax,signals_name:np.array([str])): # type: ignore
    """Band power stats of all ANC recs to be shown as further analysis"""
    NUM_PLOTS = 2
    fig, ax = plt.subplots(NUM_PLOTS, 1, figsize=(8, 6), constrained_layout=True)

    # Figure-level title
    fig.suptitle(f"Band Power between {fmin} to {fmax} — Before vs After", fontsize=14)

    # Plot 1
    ax[0].plot(signals_name, band_power_before, marker='o',color="green")
    ax[0].set_ylabel('Band Power (dB)')
    ax[0].set_title(f'Before ANC-SNR: {fmin} – {fmax} Hz')
    ax[0].tick_params(axis='x', rotation=45)

    # Plot 2
    ax[1].plot(signals_name, band_power_after, marker='o', color='orange')
    ax[1].set_ylabel('Band Power (dB)')
    ax[1].set_title(f'After ANC-SNR: {fmin} – {fmax} Hz')
    ax[1].tick_params(axis='x', rotation=45)
    mean_before,std_before=mean_std(band_power_before)
    mean_after,std_after=mean_std(band_power_after)
    
    if (len(band_power_before>1)):
        fig.text(0.5, 0.03, 
            f"Mean BP SNR before ANC: {mean_before:.2f} dB | Std: {std_before:.2f} dB", 
            ha='center', fontsize=10)


        fig.text(0.5, 0.01, 
            f"Mean BP SNR after ANC: {mean_after:.2f} dB | Std: {std_after:.2f} dB", 
            ha='center', fontsize=10)
    else:
            fig.text(0.5, 0.03, 
            f"Mean BP SNR before ANC: {mean_before:.2f} dB", 
            ha='center', fontsize=10)

            fig.text(0.5, 0.03, 
            f"Mean BP SNR after ANC: {mean_before:.2f} dB", 
            ha='center', fontsize=10)


    plt.tight_layout()
    plt.show()