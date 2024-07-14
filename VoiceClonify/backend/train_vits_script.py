import logging
import os
from datetime import datetime
from trainer import Trainer, TrainerArgs
from TTS.tts.configs.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import Vits, VitsAudioConfig
from TTS.tts.utils.text.tokenizer import TTSTokenizer
from TTS.utils.audio import AudioProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def train_vits(params):
    try:
        logger.info("Starting training process with parameters: %s", params)
        
        base_output_path = "C:\\Users\\krist\\Documents\\GitHub\\VoiceClonify\\VoiceClonify\\backend\\output"
        run_name = params.get("run_name", "vits_ljspeech")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(base_output_path, f"{run_name}_{timestamp}")
        
        os.makedirs(output_path, exist_ok=True)
        
        dataset_config = BaseDatasetConfig(
            formatter="ljspeech", 
            meta_file_train=params.get("datasets", "metadata.csv"), 
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

        # INITIALIZE THE AUDIO PROCESSOR
        # Audio processor is used for feature extraction and audio I/O.
        # It mainly serves to the dataloader and the training loggers.
        ap = AudioProcessor.init_from_config(config)
        
        # INITIALIZE THE TOKENIZER
        # Tokenizer is used to convert text to sequences of token IDs.
        # config is updated with the default characters if not defined in the config.
        tokenizer, config = TTSTokenizer.init_from_config(config)
        
        # LOAD DATA SAMPLES
        # Each sample is a list of ```[text, audio_file_path, speaker_name]```
        # You can define your custom sample loader returning the list of samples.
        # Or define your custom formatter and pass it to the `load_tts_samples`.
        # Check `TTS.tts.datasets.load_tts_samples` for more details.
        train_samples, eval_samples = load_tts_samples(
            dataset_config,
            eval_split=True,
            eval_split_max_size=config.eval_split_max_size,
            eval_split_size=config.eval_split_size,
        )
        
        # init model
        model = Vits(config, ap, tokenizer, speaker_manager=None)
        
        # init the trainer and 🚀
        trainer = Trainer(
            TrainerArgs(),
            config,
            output_path,
            model=model,
            train_samples=train_samples,
            eval_samples=eval_samples,
        )
        trainer.fit()
        
    except Exception as e:
        logger.error("Error during training: %s", e, exc_info=True)

from multiprocessing import freeze_support
if __name__ == '__main__':
    freeze_support()
