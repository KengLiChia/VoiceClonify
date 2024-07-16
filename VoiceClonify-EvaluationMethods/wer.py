import whisper
import jiwer
import argparse

def transcribe_audio_with_whisper(audio_file, model_name="base"):
    model = whisper.load_model(model_name)
    result = model.transcribe(audio_file, language="en")
    return result["text"]

def calculate_wer(reference_text, hypothesis_text):
    error = jiwer.wer(reference_text, hypothesis_text)
    return error

def main(original_file, generated_file):
    reference_transcription = transcribe_audio_with_whisper(original_file)
    hypothesis_transcription = transcribe_audio_with_whisper(generated_file)
    
    print(f"Reference Transcription: {reference_transcription}")
    print(f"Hypothesis Transcription: {hypothesis_transcription}")
    
    wer_value = calculate_wer(reference_transcription, hypothesis_transcription)
    print(f"Word Error Rate (WER): {wer_value}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calculate WER between two audio files")
    parser.add_argument("original", type=str, help="Path to the original audio file")
    parser.add_argument("generated", type=str, help="Path to the generated audio file")
    
    args = parser.parse_args()
    main(args.original, args.generated)
