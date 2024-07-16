import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
from pesq import pesq
import librosa
import librosa.display

def preprocess_audio(filename, target_rate=16000):
    rate, data = wavfile.read(filename)
    if rate != target_rate:
        data = librosa.resample(data.astype(float), orig_sr=rate, target_sr=target_rate)
        rate = target_rate
    data = librosa.util.normalize(data.astype(np.int16))  # Normalize amplitude and convert to int16
    return rate, data

def calculate_pesq(original_file, generated_file, mode='wb'):
    original_rate, original = preprocess_audio(original_file)
    generated_rate, generated = preprocess_audio(generated_file)
    
    if original_rate != generated_rate:
        raise ValueError("Sample rates of original and generated files do not match.")
    
    pesq_score = pesq(original_rate, original, generated, mode)
    return pesq_score

def plot_waveforms_and_spectrograms(original_file, generated_file, sr=16000):
    _, original = preprocess_audio(original_file, target_rate=sr)
    _, generated = preprocess_audio(generated_file, target_rate=sr)
    
    plt.figure(figsize=(12, 8))

    # Plot waveforms
    plt.subplot(2, 2, 1)
    plt.title(f"Waveform of Original Speech: {original_file}")
    librosa.display.waveshow(original, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    plt.subplot(2, 2, 2)
    plt.title(f"Waveform of Generated Speech: {generated_file}")
    librosa.display.waveshow(generated, sr=sr)
    plt.xlabel("Time")
    plt.ylabel("Amplitude")

    # Plot spectrograms
    plt.subplot(2, 2, 3)
    plt.title(f"Spectrogram of Original Speech: {original_file}")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(original)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.subplot(2, 2, 4)
    plt.title(f"Spectrogram of Generated Speech: {generated_file}")
    D = librosa.amplitude_to_db(np.abs(librosa.stft(generated)), ref=np.max)
    librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.xlabel("Time")
    plt.ylabel("Frequency")

    plt.tight_layout()
    plt.show()

def main(original_file, generated_file):
    pesq_value = calculate_pesq(original_file, generated_file)
    print(f"Perceptual Evaluation of Speech Quality (PESQ): {pesq_value}")

    plot_waveforms_and_spectrograms(original_file, generated_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PESQ between two audio files")
    parser.add_argument("original", type=str, help="Path to the original audio file")
    parser.add_argument("generated", type=str, help="Path to the generated audio file")
    
    args = parser.parse_args()
    main(args.original, args.generated)
