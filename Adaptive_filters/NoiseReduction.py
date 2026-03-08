import numpy as np
import librosa
import soundfile as sf

# -------------------------
# Utilities
# -------------------------

def to_mono(y):
    if y.ndim > 1:
        return np.mean(y, axis=1)
    return y

def resample(y, sr_in, sr_out):
    if sr_in == sr_out:
        return y
    return librosa.resample(y, orig_sr=sr_in, target_sr=sr_out)

# -------------------------
# DNS-4 (SpeechBrain)
# -------------------------

# def dns_denoise(y, sr):
#     try:
#         import torch
#         import torchaudio
#         from speechbrain.pretrained import SepformerSeparation

#         TARGET_SR = 16000
#         y_rs = resample(y, sr, TARGET_SR)

#         model = SepformerSeparation.from_hparams(
#             source="speechbrain/sepformer-dns4-16k-enhancement",
#             savedir="models/dns4"
#         )

#         with torch.no_grad():
#             wav = torch.from_numpy(y_rs).float().unsqueeze(0)
#             enhanced = model.enhance_batch(wav)

#         return enhanced.numpy(), TARGET_SR

#     except Exception as e:
#         return None, None

# -------------------------
# RNNoise
# -------------------------

def rnnoise_denoise(y, sr):
    try:
        import noisereduce as nr

        TARGET_SR = 48000
        y_rs = resample(y, sr, TARGET_SR)

        y_dn = nr.reduce_noise(y=y, sr=sr)

        return y_dn, TARGET_SR

    except Exception:
        return None, None

# -------------------------
# WebRTC NS (optional fallback)
# -------------------------

# def webrtc_denoise(y, sr):
#     try:
#         import webrtc_noise_suppression as ns

#         y_int16 = np.int16(np.clip(y, -1, 1) * 32767)
#         suppressor = ns.NoiseSuppressor(sr, ns.Level.HIGH)
#         y_dn = suppressor.process(y_int16)

#         return y_dn.astype(np.float32) / 32768.0, sr

#     except Exception:
#         return None, None

# # -------------------------
# # Master selector
# # -------------------------

# def smart_denoise(y, sr):
#     """
#     Automatically choose best denoiser.
#     Input:
#         y  : mono float32 audio [-1, 1]
#         sr : any sample rate
#     Output:
#         y_clean, sr_out
#     """

#     y = to_mono(y)

#     # 1. Best offline quality
#     y_dn, sr_dn = dns_denoise(y, sr)
#     if y_dn is not None:
#         return y_dn, sr_dn

#     # 2. Real-time grade but strong
#     y_dn, sr_dn = rnnoise_denoise(y, sr)
#     if y_dn is not None:
#         return y_dn, sr_dn

#     # 3. Conservative fallback
#     y_dn, sr_dn = webrtc_denoise(y, sr)
#     if y_dn is not None:
#         return y_dn, sr_dn

#     # 4. No denoising possible
#     return y, sr
