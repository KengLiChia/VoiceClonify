import librosa
import numpy as np
import pyworld
import pysptk
import argparse
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import soundfile as sf

def preprocess_audio(wav_file, sample_rate=22050):
    wav, _ = librosa.load(wav_file, sr=sample_rate, mono=True)
    wav, _ = librosa.effects.trim(wav)  # Trim silence
    wav = librosa.util.normalize(wav)  # Normalize amplitude
    return wav

def wav2mcep_simple(wav, sample_rate=22050, frame_period=5.0, alpha=0.65, fft_size=512):
    try:
        f0, sp, ap = pyworld.wav2world(wav.astype(np.double), fs=sample_rate, frame_period=frame_period, fft_size=fft_size)
        mcep = pysptk.sp2mc(sp, order=13, alpha=alpha)
        return mcep
    except Exception as e:
        print(f"Error processing wav: {e}")
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
    
    ref_wav = preprocess_audio(ref_audio_file, sample_rate)
    syn_wav = preprocess_audio(syn_audio_file, sample_rate)
    
    ref_mcep = wav2mcep_simple(ref_wav, sample_rate, frame_period)
    syn_mcep = wav2mcep_simple(syn_wav, sample_rate, frame_period)
    
    if ref_mcep is None or syn_mcep is None:
        return None, ref_wav, syn_wav
    
    _, path = fastdtw(ref_mcep[:, 1:], syn_mcep[:, 1:], dist=euclidean)
    frames_tot, min_cost_tot = calculate_mcd_distance(ref_mcep, syn_mcep, path)
    mean_mcd = log_spec_dB_const * min_cost_tot / frames_tot
    
    return mean_mcd, ref_wav, syn_wav

def plot_waveforms(real, generated, real_path, generated_path):
    plt.figure(figsize=(12, 6))

    plt.subplot(2, 1, 1)
    plt.title(f"Real Speech")
    plt.plot(real)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 1, 2)
    plt.title(f"Generated Speech")
    plt.plot(generated)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()

def main(real_path, generated_path):
    real = preprocess_audio(real_path)
    generated = preprocess_audio(generated_path)

    mcd, ref_wav, syn_wav = calculate_mcd(real_path, generated_path)
    if mcd is not None:
        print(f"MCD: {mcd}")
    else:
        print("MCD calculation failed due to an error in processing the audio files.")

    # plot_waveforms(real, generated, real_path, generated_path)

    # Save the trimmed and normalized audio files for listening
    sf.write('trimmed_normalized_real.wav', ref_wav, 22050)
    sf.write('trimmed_normalized_generated.wav', syn_wav, 22050)
    print(f"Trimmed and normalized audio files saved as 'trimmed_normalized_real.wav' and 'trimmed_normalized_generated.wav'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MCD and plot waveforms.")
    parser.add_argument("real_path", type=str, help="Path to the real speech file.")
    parser.add_argument("generated_path", type=str, help="Path to the generated speech file.")

    args = parser.parse_args()
    main(args.real_path, args.generated_path)

# python .\calculate_mcd.py .\data\original.wav .\data\generated.wav