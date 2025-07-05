import numpy as np
import soundfile as sf
from pathlib import Path
import scipy.signal as sp
import matplotlib.pyplot as plt

"""helping functions page and class to the signal"""


class SigArgs:
    """Sig arguments class to descibe signal properties to a better output and assesment"""
    def __init__(self, sig, fs, name):
        """initialize a sig args object"""
        self._sig_array = sig
        self._fs = fs
        self._sig_name = name

    """Class decorators and properties"""
    @property
    def sig_array(self):
        return self._sig_array

    @sig_array.setter
    def sig_array(self, value):
        self._sig_array = value

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, value):
        self._fs = value

    @property
    def name(self):
        return self._sig_name

    @name.setter
    def name(self, value):
        self._sig_name = value

def resample_fs(sig,fs_old,fs_new):
    """Function that resample a signal based on new sample rate by using scipy resample"""
    num_samples = int(len(sig) * fs_new / fs_old)
    resampled_sig = sp.resample(sig, num_samples)
    return resampled_sig,fs_new

def common_length(sig1,sig2):
    """Function that returns the common length of 2 signals"""
    min_len = min(len(sig1), len(sig2))
    return min_len

def create_folder(path: Path, folder_name: str):
    """
    Function using Pathlib library to make new folder if needed.
    path / folder_name: creates a Path object for the new folder.

    mkdir(parents=True, exist_ok=True):

    parents=True: creates intermediate folders if they don’t exist.

    exist_ok=True: avoids error if the folder already exists.

    """
    new_folder = path / folder_name
    new_folder.mkdir(parents=True, exist_ok=True)#checks if some folder exits in that directory
    return new_folder
    
def generate_full_sig(sig_object:SigArgs,noise_object:SigArgs,noise_lin_amp:float=3.0):
    """
    Gets Signal Object and adds to it Noise Object, final outcome: X(t)+N(t).
    the function cares that it will be at the same length and it will be resampled
    returns the new signal object!
    """
    noise=noise_object._sig_array
    fs_noise=noise_object.fs

    sig=sig_object._sig_array
    fs_sig=sig_object.fs
    
    if(noise.ndim>1):
        noise = noise.mean(axis=1)
        noise_object.sig_array=noise
    if(sig.ndim>1):    
        sig=sig.mean(axis=1)
        sig_object.sig_array=sig

    if fs_sig>fs_noise:
        sig,fs1=resample_fs(sig,fs_old=fs_sig,fs_new=fs_noise)
        sig_object.sig_array=sig
        sig_object.fs=fs1

    elif fs_sig<fs_noise:
        noise,fs2=resample_fs(sig,fs_old=fs_noise,fs_new=fs_sig)
        noise_object.sig_array=noise
        noise_object.fs=fs2

        
    coomon_len=common_length(sig,noise)
    sig=sig[:coomon_len]
    noise=noise[:coomon_len]
    noise=noise[:coomon_len]*noise_lin_amp
    final_sig=sig+noise
    sig_object.sig_array=sig
    noise_object.sig_array=noise#must rethink again
    return SigArgs(final_sig,noise_object.fs,f"{sig_object.name}_with_{noise_object.name}")



