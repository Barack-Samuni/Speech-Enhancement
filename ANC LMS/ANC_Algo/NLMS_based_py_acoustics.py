from pyroomacoustics.adaptive import NLMS
import numpy as np
import soundfile as sf
import scipy.signal as sp
import matplotlib.pyplot as plt
from helping_methods_classes import resample_fs

def NLMS_calculation(total_sig, noise, fs1, fs2, txt_file:str, filter_len: int = 128, mu: float = 0.01):
    """NLMS adaptive noise cancellation"""
    if fs1>fs2:
        sig,fs1=resample_fs(sig,fs_old=fs1,fs_new=fs2)

    elif fs1<fs2:
        noise,fs2=resample_fs(sig,fs_old=fs2,fs_new=fs1)

    # Use only first channel if stereo
    if total_sig.ndim > 1:
        total_sig = total_sig.mean(axis=1)
    if noise.ndim > 1:
        noise = noise.mean(axis=1)

    # Match lnegths
    min_len = min(len(total_sig), len(noise))
    total_sig = total_sig[:min_len]
    noise = noise[:min_len]

    # Init NLMS filter
    l = NLMS(filter_len,mu)
    e = np.zeros(min_len)

    # Adaptive filtering loop- update weights for each sample
    for n in range(min_len):
        l.update(noise[n], total_sig[n])
        y = np.dot(l.w, l.x)
        e[n] = total_sig[n] - y  # cleaned signal

    # Save output
    # sf.write(r"C:\Users\galon\Documents\projects\Wavs\exp1\Anc_cleaned_output1.wav", e, fs1)
    with open(txt_file, 'a') as file:
         file.write(f"filter len:{filter_len},mu: {mu}")
         file.write("\n")

    print("Anc of full signal and noise Completed")
    
    return e,fs1

