from pathlib import Path
import soundfile as sf
import numpy as np
from helping_methods_classes import SigArgs,generate_full_sig,create_folder
from NLMS_based_py_acoustics import NLMS_calculation
from plots import plot_spectograms_of_all,signal_noise_comparison,bandpower_statics

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
    except ValueError:
        print("invalid value- must be integer")
        raise ValueError
           
    if not (0 <= index < n):
      print("Index must be in the right limits")
      raise IndexError
        
      
    else:
      sig, fs = sf.read(wav_files[index])
      return SigArgs(sig, fs, wav_files[index].stem)


def save_file_in_folder(folder_path:Path,sig:SigArgs):
  file_name=sig.name+".wav"
  file_path_saving=folder_path/file_name
  data_sig=sig.sig_array
  fs_sig=sig.fs
  sf.write(folder_path,data_sig,fs_sig)
  print("Full signal is saved")

def running_NLMS(list_noise: list[SigArgs], sig: SigArgs, folder_total_sigs: Path, folder_anc_sigs: Path):
    data_bp_before = np.array([])
    data_bp_after = np.array([])
    sig_names_array = np.array([])
    fmin = 300
    fmax = 5000

    for noise in list_noise:
        full_signal = generate_full_sig(sig_object=sig, noise_object=noise)
        save_file_in_folder(folder_total_sigs, full_signal)
        text_full_path = f"{folder_anc_sigs}/{full_signal.name}.txt"

        with open(text_full_path, 'w') as file:
            file.write(f"Anc with {sig.name}-NLMS Method\n")
            file.write(f"Signal fs: {full_signal.fs}\n")

        anc_signal, fs1 = NLMS_calculation(
            total_sig=full_signal.sig_array,
            noise=noise.sig_array,
            fs1=full_signal.fs,
            fs2=noise.fs,
            txt_file=text_full_path
        )

        sig_names_array = np.append(sig_names_array, full_signal.name)
        anc_object = SigArgs(anc_signal, fs1, f"Anc_{full_signal.name}")
        save_file_in_folder(folder_anc_sigs, anc_object)
        plot_spectograms_of_all(
            total_sig=full_signal.sig_array,
            noise=noise.sig_array,
            cleaned_sig=anc_object.sig_array
        )

        _, _, _, bp_snr_before, bp_snr_after = signal_noise_comparison(
            sig=sig.sig_array,
            noise=noise.sig_array,
            anc_sig=anc_signal,
            fs=fs1,
            fmin=fmin,
            fmax=fmax
        )

        data_bp_before = np.append(data_bp_before, bp_snr_before)
        data_bp_after = np.append(data_bp_after, bp_snr_after)

        with open(text_full_path, 'a') as file:
            file.write("based on welch method of PSDs:\n")
            file.write(f"SNR band_power before ANC {bp_snr_before}\n")
            file.write(f"SNR band_power after ANC {bp_snr_after}\n")

    bandpower_statics(
        band_power_before=data_bp_before,
        band_power_after=data_bp_after,
        fmin=fmin,
        fmax=fmax,
        signals_name_before=sig_names_array,
        signals_name_after=sig_names_array
    )

def main():
  root_folder_path=r"C:\Users\galon\Documents\projects\Wavs"
  #USER INPUT of audio recs path
  folder_path_noise=Path(f"{root_folder_path}/Noises")
  folder_path_sig=Path(f"{root_folder_path}/Signals")#Giving the signal Folder

  noise_list=get_noises_by_folder(folder=folder_path_noise)
  original_sig=get_one_sig(folder_sig=folder_path_sig)
  folder_total_sigs=(create_folder(Path(root_folder_path),"Total_noised_Sigs"))
  folder_anc_sigs=(create_folder(Path(root_folder_path),"Anc_signals"))
  running_NLMS(noise_list,original_sig,folder_total_sigs,folder_anc_sigs)



if __name__=="__main__":
    main()