import librosa
import numpy as np
import pyworld
import pysptk
import argparse
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def load_wav(wav_file, sample_rate):
    wav, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
    return wav

def wav2mcep_numpy(loaded_wav, sample_rate, frame_period, alpha=0.65, fft_size=512):
    _, sp, _ = pyworld.wav2world(loaded_wav.astype(np.double), fs=sample_rate, frame_period=frame_period, fft_size=fft_size)
    mcep = pysptk.sptk.mcep(sp, order=13, alpha=alpha, maxiter=0, etype=1, eps=1.0E-8, min_det=0.0, itype=3)
    return mcep

def calculate_mcd_distance(ref_mcep, syn_mcep, path):
    pathx = list(map(lambda l: l[0], path))
    pathy = list(map(lambda l: l[1], path))
    ref_mcep, syn_mcep = ref_mcep[pathx], syn_mcep[pathy]
    frames_tot = ref_mcep.shape[0]
    diff = ref_mcep - syn_mcep
    min_cost_tot = np.sqrt((diff * diff).sum(-1)).sum()
    return frames_tot, min_cost_tot

def calculate_mcd(ref_audio_file, syn_audio_file):
    sample_rate = 22050
    frame_period = 5.0
    log_spec_dB_const = 10.0 / np.log(10.0) * np.sqrt(2.0)
    
    loaded_ref_wav = load_wav(ref_audio_file, sample_rate)
    loaded_syn_wav = load_wav(syn_audio_file, sample_rate)
    
    ref_mcep = wav2mcep_numpy(loaded_ref_wav, sample_rate, frame_period)
    syn_mcep = wav2mcep_numpy(loaded_syn_wav, sample_rate, frame_period)
    
    _, path = fastdtw(ref_mcep[:, 1:], syn_mcep[:, 1:], dist=euclidean)
    frames_tot, min_cost_tot = calculate_mcd_distance(ref_mcep, syn_mcep, path)
    mean_mcd = log_spec_dB_const * min_cost_tot / frames_tot
    
    return mean_mcd

def plot_waveforms(real, generated, real_path, generated_path):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title(f"Real Speech: {real_path}")
    plt.plot(real)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.title(f"Generated Speech: {generated_path}")
    plt.plot(generated)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def main(real_path, generated_path):
    real, sr = librosa.load(real_path, sr=22050)
    generated, sr = librosa.load(generated_path, sr=22050)

    mcd = calculate_mcd(real_path, generated_path)
    print(f"MCD: {mcd}")

    plot_waveforms(real, generated, real_path, generated_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MCD and plot waveforms.")
    parser.add_argument("real_path", type=str, help="Path to the real speech file.")
    parser.add_argument("generated_path", type=str, help="Path to the generated speech file.")

    args = parser.parse_args()
    main(args.real_path, args.generated_path)
