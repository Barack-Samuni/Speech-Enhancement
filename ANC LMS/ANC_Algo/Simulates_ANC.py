from pathlib import Path
import soundfile as sf
import numpy as np
from helping_methods_classes import SigArgs


def get_noises_by_folder(folder:Path):
      """
      *sf.read(wav_file) unpacks the tuple (noise_array, fs)

      wav_file.stem gives the file name without extension

      put in sig Args class and then do it to each noise file
      """
      if len(list(folder.glob("*wav"))) == 0:
         raise FileNotFoundError(f"No .wav files found in the folder: {folder}")
        
      
      return [SigArgs(*sf.read(n), n.stem) for n in folder.glob("*.wav")]   
    

def printfiles(wav_files: list):
    """print all wav files in specified folder"""
    for i, f in enumerate(wav_files):
        print(f"[{i}] {f.name}")

    print("\n")

def get_one_sig(folder_sig:Path):
    """returns the chosen cleaned signal to do on it ANC"""
    wav_files=list(folder_sig.glob("*.wav"))
    n=len(wav_files)
    if n == 0:
        print("No .wav files found in the folder.")
        raise FileNotFoundError(f"No .wav files found in the folder: {folder_sig}")

    printfiles(wav_files)
    try:
      index=int(input(f"Enter index of desired signal between 0 to {n-1}"))
    except ValueError as e :
        print("invalid value- must be integer")
        raise e
           
    if not (0 <= index < n):
      print("Index must be in the right limits")
      raise IndexError
        
      
    else:
      sig, fs = sf.read(wav_files[index])
      return SigArgs(sig, fs, wav_files[index].stem)


def main():
  folder_path_noise=Path(r"C:\Users\Omer\Documents\Projects\WAVS\ANC WAVS\Noises")#USER INPUT OF NOISES FOLDER
  folder_path_sig=Path(r"C:\Users\galon\Documents\projects\Wavs\Signals")#Giving the signal Folder

  noise_list=get_noises_by_folder(folder=folder_path_noise)
  sig=get_one_sig(folder_sig=folder_path_sig)
   
  



if __name__=="__main__":
    main()