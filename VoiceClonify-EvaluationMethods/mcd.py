import numpy as np
from scipy.io import wavfile
from python_speech_features import mfcc
from scipy.spatial.distance import euclidean
import argparse
import librosa

def calculate_mcd(original, generated, num_mel_coefficients=13):
    original_mfcc = mfcc(original, numcep=num_mel_coefficients)
    generated_mfcc = mfcc(generated, numcep=num_mel_coefficients)
    
    # Align the lengths of the MFCC sequences
    min_length = min(len(original_mfcc), len(generated_mfcc))
    original_mfcc = original_mfcc[:min_length]
    generated_mfcc = generated_mfcc[:min_length]
    
    # Compute the Euclidean distance for each frame
    distances = [euclidean(original_mfcc[i], generated_mfcc[i]) for i in range(min_length)]
    
    # Calculate the average distance (MCD) with scaling factor
    scaling_factor = 10 / np.log(10)
    mcd = scaling_factor * np.mean(distances)
    return mcd

def load_wav(filename, target_rate=22050): # this is my 22050 Hz generated
    rate, data = wavfile.read(filename)
    if rate != target_rate:
        data = librosa.resample(data.astype(float), orig_sr=rate, target_sr=target_rate) # resample to 22050 Hz
    return data

def main(original_file, generated_file):
    original = load_wav(original_file)
    generated = load_wav(generated_file)
    
    mcd_value = calculate_mcd(original, generated)
    print(f"Mel Cepstral Distortion (MCD): {mcd_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate MCD between two audio files")
    parser.add_argument("original", type=str, help="Path to the original audio file")
    parser.add_argument("generated", type=str, help="Path to the generated audio file")
    
    args = parser.parse_args()
    main(args.original, args.generated)
