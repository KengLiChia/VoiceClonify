from TTS.api import TTS
import gradio as gr

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont('Inter')],
    font_mono=[gr.themes.GoogleFont('Fira Mono')],
)

with gr.Blocks(theme=theme, title="Voice Clonify") as demo:
    gr.Markdown(
        """
        # Voice Clonify
        Clone your very own voices for text to speech generation 🗣️.
        
        Supported languages: Arabic: ar, Brazilian Portuguese: pt , Mandarin Chinese: zh-cn, Czech: cs, Dutch: nl, English: en, French: fr, German: de, Italian: it, Polish: pl, Russian: ru, Spanish: es, Turkish: tr, Japanese: ja, Korean: ko, Hungarian: hu, Hindi: hi
        """
    )

    with gr.Tab("Train XTTS"):
        # Components for "Train XTTS" tab
        text_prompt = gr.Textbox(label="Text Prompt", placeholder="Hi there, I'm a voice clone!", info="Enter the text you want to generate speech for.")
        language = gr.Dropdown(label="Language", choices=["ar", "pt", "zh-cn", "cs", "nl", "en", "fr", "de", "it", "pl", "ru", "es", "tr", "ja", "ko", "hu", "hi"], info="Select the language for the generated speech.")
        speaker_wav_file = gr.Audio(label="Reference Audio", type="filepath", format=["wav"])
        output_file_path = gr.Textbox(label="Output File Path", placeholder="output.wav")
        
        # Submit Button
        submit_button_xtts = gr.Button("Submit XTTS", variant="primary")
     

        def run_voice_clonify_xtts():
            print(text_prompt)

            # print(language_xtts_value)
            # output_file_path_value = output_file_path.value

            # # Create TTS object
            # tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

            # # Generate speech by cloning a voice using default settings
            # tts.tts_to_file(text="Hi Testing", file_path=output_file_path_value, speaker_wav=[speaker_wav_path], language=language_xtts_value)

            # output_text_xtts = f"Speech generated and saved to {output_file_path_value}"
            # return output_text_xtts
            
            submit_button_xtts.click(run_voice_clonify_xtts, inputs=[text_prompt, language, speaker_wav_file, output_file_path], outputs=[output_xtts])
    with gr.Tab("Train VITTS"):
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    run_name = gr.Textbox(label="Run Name", placeholder="vits_ljspeech")
                    batch_size = gr.Slider(label="Batch Size", minimum=0, maximum=100, step=1)
                    eval_batch_size = gr.Slider(label="Eval Batch Size", minimum=0, maximum=100, step=1)
                    batch_group_size = gr.Slider(label="Batch Group Size", minimum=0, maximum=100, step=1)
                    num_loader_workers = gr.Slider(label="Num Loader Workers", minimum=0, maximum=100, step=1)

                with gr.Column(scale=1, min_width=600):
                    run_eval = gr.Checkbox(label="Run Eval", value=False)
                    num_eval_loader_workers = gr.Slider(label="Num Eval Loader Workers", minimum=0, maximum=100, step=1)
                    test_delay_epochs = gr.Textbox(label="Test Delay Epochs", placeholder="-1")
                    epochs = gr.Textbox(label="Epochs", placeholder="25000")
                    text_cleaner = gr.Textbox(label="Text Cleaner", placeholder="english_cleaners", interactive=False)

            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    use_phonemes = gr.Checkbox(label="Use Phonemes", value=False)
                    phoneme_language = gr.Textbox(label="Phoneme Language", placeholder="en-us", interactive=False)
                    phoneme_cache_path = gr.Textbox(label="Phoneme Cache Path", placeholder="phoneme_cache", interactive=False)

                with gr.Column(scale=1, min_width=600):
                    compute_input_seq_cache = gr.Checkbox(label="Compute Input Seq Cache", value=False)
                    print_step = gr.Textbox(label="Print Step", placeholder="25")
                    print_eval = gr.Checkbox(label="Print Eval", value=False)
                    mixed_precision = gr.Checkbox(label="Mixed Precision", value=False)

            output_path = gr.Textbox(label="Output Path", placeholder="/path/to/output")
            datasets = gr.Textbox(label="Datasets", placeholder="metadata.csv")
            test_sentences = gr.Textbox(label="Test Sentences", placeholder="Sentence 1\nSentence 2\nSentence 3")

            cudnn_benchmark = gr.Checkbox(label="CUDNN Benchmark", value=False)

            with gr.Row():
                button = gr.Button("Submit", variant="primary")

            output = gr.Textbox(label="Output")

            def run_voice_clonify():
                # Get values from Gradio components
                config_params = {
                    "run_name": run_name.value,
                    "batch_size": int(batch_size.value),
                    "eval_batch_size": int(eval_batch_size.value),
                    "batch_group_size": int(batch_group_size.value),
                    "num_loader_workers": int(num_loader_workers.value),
                    "num_eval_loader_workers": int(num_eval_loader_workers.value),
                    "run_eval": run_eval.value,
                    "test_delay_epochs": int(test_delay_epochs.value),
                    "epochs": int(epochs.value),
                    "text_cleaner": text_cleaner.value,
                    "use_phonemes": use_phonemes.value,
                    "phoneme_language": phoneme_language.value,
                    "phoneme_cache_path": phoneme_cache_path.value,
                    "compute_input_seq_cache": compute_input_seq_cache.value,
                    "print_step": int(print_step.value),
                    "print_eval": print_eval.value,
                    "mixed_precision": mixed_precision.value,
                    "output_path": output_path.value,
                    "datasets": [{"formatter": "ljspeech", "meta_file_train": datasets.value, "path": "/path/to/dataset"}],
                    "cudnn_benchmark": cudnn_benchmark.value,
                    "test_sentences": test_sentences.value.split("\n"),
                }

                # Execute your script with the provided parameters
                # Replace the following line with your script execution
                output_text = f"Running with parameters: {config_params}"
                return output_text

            button.click(run_voice_clonify, outputs=[output])

demo.launch(share=True)
