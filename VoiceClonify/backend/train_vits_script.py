import os
import time
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor
from multiprocessing import Queue

def find_metadata_file(dataset_path):
    print("Looking for metadata file in:", dataset_path)
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.csv'):
            metadata_path = os.path.join(dataset_path, file_name)
            print("Found metadata file:", metadata_path)
            return metadata_path
    raise FileNotFoundError("No metadata CSV file found in the dataset directory.")

def train_vits(params, queue):
    base_output_path = os.path.dirname(os.path.abspath(__file__))
    run_name = params.get("run_name", "vits_ljspeech")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(base_output_path, f"{run_name}_{timestamp}")
    
    os.makedirs(output_path, exist_ok=True)
    
    dataset_path = os.path.abspath(params.get("datasets"))
    dataset_config = BaseDatasetConfig(
        formatter="ljspeech", 
        meta_file_train=find_metadata_file(dataset_path), 
        path=params.get("dataset_path", "C:\\Users\\krist\\Documents\\GitHub\\VoiceClonify\\dataset\\LJSpeech-1.1")
    )
    audio_config = VitsAudioConfig(
        sample_rate=22050, win_length=1024, hop_length=256, num_mels=80, mel_fmin=0, mel_fmax=None
    )
    
    config = VitsConfig(
        audio=audio_config,
        run_name=run_name,
        batch_size=params.get("batch_size", 32),
        eval_batch_size=params.get("eval_batch_size", 16),
        batch_group_size=params.get("batch_group_size", 5),
        num_loader_workers=params.get("num_loader_workers", 8),
        num_eval_loader_workers=params.get("num_eval_loader_workers", 4),
        run_eval=params.get("run_eval", True),
        test_delay_epochs=params.get("test_delay_epochs", -1),
        epochs=params.get("epochs", 6072),
        save_step=10000,
        text_cleaner=params.get("text_cleaner", "english_cleaners"),
        use_phonemes=params.get("use_phonemes", True),
        phoneme_language=params.get("phoneme_language", "en-us"),
        phoneme_cache_path=os.path.join(output_path, "phoneme_cache"),
        compute_input_seq_cache=params.get("compute_input_seq_cache", True),
        print_step=params.get("print_step", 25),
        print_eval=params.get("print_eval", True),
        mixed_precision=params.get("mixed_precision", True),
        output_path=output_path,
        datasets=[dataset_config],
        cudnn_benchmark=params.get("cudnn_benchmark", False),
        test_sentences=params.get("test_sentences", [
            "The quick brown fox jumps over the lazy dog.",
        ]),
        save_n_checkpoints=10,
        phonemizer="espeak"
    )

    ap = AudioProcessor.init_from_config(config)
    tokenizer, config = TTSTokenizer.init_from_config(config)
    train_samples, eval_samples = load_tts_samples(
        dataset_config,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )

    model = Vits(config, ap, tokenizer, speaker_manager=None)
    trainer = Trainer(
        TrainerArgs(),
        config,
        output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
    )
    trainer.fit()
    queue.put(output_path)  # Put the unique output_path in the queue for Flask to retrieve
    print(output_path)

if __name__ == '__main__':
    from multiprocessing import freeze_support, Queue
    freeze_support()
    queue = Queue()
    params = {
        "run_name": "vits_ljspeech",
        "batch_size": 32,
        "eval_batch_size": 16,
        "batch_group_size": 5,
        "num_loader_workers": 8,
        "num_eval_loader_workers": 4,
        "run_eval": True,
        "test_delay_epochs": -1,
        "epochs": 6072,
        "text_cleaner": "english_cleaners",
        "use_phonemes": True,
        "phoneme_language": "en-us",
        "phoneme_cache_path": "phoneme_cache",
        "compute_input_seq_cache": True,
        "print_step": 25,
        "print_eval": True,
        "mixed_precision": True,
        "output_path": "output_path",
        "datasets": "metadata.csv",
        "cudnn_benchmark": False,
        "test_sentences": ["The quick brown fox jumps over the lazy dog."],
        "dataset_path": "C:\\Users\\krist\\Documents\\GitHub\\VoiceClonify\\dataset\\LJSpeech-1.1"
    }
    train_vits(params, queue)
    print("Output path:", queue.get())