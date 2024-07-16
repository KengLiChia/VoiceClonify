import argparse
import numpy as np
from scipy.io import wavfile
from pesq import pesq
import librosa

def load_wav(filename, target_rate=16000): # I have to resample both files to 16000 Hz cos PEQ standard
    rate, data = wavfile.read(filename)
    if rate != target_rate:
        data = librosa.resample(data.astype(float), orig_sr=rate, target_sr=target_rate)
        rate = target_rate
    return rate, data

def calculate_pesq(original_file, generated_file, mode='wb'):
    original_rate, original = load_wav(original_file)
    generated_rate, generated = load_wav(generated_file)
    
    if original_rate != generated_rate:
        raise ValueError("Sample rates of original and generated files do not match.")
    
    pesq_score = pesq(original_rate, original, generated, mode)
    return pesq_score

def main(original_file, generated_file):
    pesq_value = calculate_pesq(original_file, generated_file)
    print(f"Perceptual Evaluation of Speech Quality (PESQ): {pesq_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate PESQ between two audio files")
    parser.add_argument("original", type=str, help="Path to the original audio file")
    parser.add_argument("generated", type=str, help="Path to the generated audio file")
    
    args = parser.parse_args()
    main(args.original, args.generated)
