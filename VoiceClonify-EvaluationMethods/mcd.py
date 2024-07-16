import librosa
import numpy as np
import pyworld
import pysptk
import argparse
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def wav2mcep_simple(wav_file, sample_rate=22050, frame_period=5.0, alpha=0.65, fft_size=512):
    try:
        wav, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
        f0, sp, ap = pyworld.wav2world(wav.astype(np.double), fs=sample_rate, frame_period=frame_period, fft_size=fft_size)
        mcep = pysptk.sp2mc(sp, order=13, alpha=alpha)
        return mcep
    except Exception as e:
        print(f"Error processing {wav_file}: {e}")
        return None

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
    
    ref_mcep = wav2mcep_simple(ref_audio_file, sample_rate, frame_period)
    syn_mcep = wav2mcep_simple(syn_audio_file, sample_rate, frame_period)
    
    if ref_mcep is None or syn_mcep is None:
        return None
    
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

def plot_mcep_features(ref_mcep, syn_mcep, ref_path, syn_path):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title(f"MCEP Features of Real Speech: {ref_path}")
    plt.imshow(ref_mcep.T, aspect='auto', origin='lower')
    plt.xlabel("Frame")
    plt.ylabel("MCEP Coefficient")

    plt.subplot(2, 1, 2)
    plt.title(f"MCEP Features of Generated Speech: {syn_path}")
    plt.imshow(syn_mcep.T, aspect='auto', origin='lower')
    plt.xlabel("Frame")
    plt.ylabel("MCEP Coefficient")

    plt.tight_layout()
    plt.show()

def main(real_path, generated_path):
    real, sr = librosa.load(real_path, sr=22050)
    generated, sr = librosa.load(generated_path, sr=22050)

    mcd = calculate_mcd(real_path, generated_path)
    if mcd is not None:
        print(f"MCD: {mcd}")
    else:
        print("MCD calculation failed due to an error in processing the audio files.")

    plot_waveforms(real, generated, real_path, generated_path)

    ref_mcep = wav2mcep_simple(real_path)
    syn_mcep = wav2mcep_simple(generated_path)

    if ref_mcep is not None and syn_mcep is not None:
        plot_mcep_features(ref_mcep, syn_mcep, real_path, generated_path)
    else:
        print("MCEP feature extraction failed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MCD and plot waveforms.")
    parser.add_argument("real_path", type=str, help="Path to the real speech file.")
    parser.add_argument("generated_path", type=str, help="Path to the generated speech file.")

    args = parser.parse_args()
    main(args.real_path, args.generated_path)
