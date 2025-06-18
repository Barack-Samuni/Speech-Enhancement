import numpy as np
import soundfile as sf
from scipy.signal import resample
def resample_fs(sig,fs_old,fs_new):
    num_samples = int(len(sig) * fs_new / fs_old)
    resampled_sig = resample(sig, num_samples)
    return resampled_sig,fs_new

def common_length(sig1,sig2):
    min_len = min(len(sig1), len(sig2))
    return min_len

sig, fs1 = sf.read(r"C:\Users\galon\Documents\projects\Wavs\Signals\races.wav")
noise, fs2=sf.read(r"C:\Users\galon\Documents\projects\Wavs\Noises\compress my life.wav")
if(noise.ndim>=1):
    noise = noise.mean(axis=1)
if(sig.ndim>=1):    
    sig=sig.mean(axis=1)
if fs1>fs2:
    sig,fs1=resample_fs(sig,fs_old=fs1,fs_new=fs2)

elif fs1<fs2:
    noise,fs2=resample_fs(sig,fs_old=fs2,fs_new=fs1)
coomon_len=common_length(sig,noise)
sig=sig[:coomon_len]
noise=noise[:coomon_len]*3
final_sig=sig+noise
file_add=r"C:\Users\galon\Documents\projects\Wavs\exp1"
sf.write(file_add+r"\full_sig_noise_sig2.wav",final_sig, fs1)
sf.write(file_add+r"\exp_sig2.wav",sig,fs1)
sf.write(file_add+r"\exp_noise2.wav",noise,fs1)
print("Succsseeded")