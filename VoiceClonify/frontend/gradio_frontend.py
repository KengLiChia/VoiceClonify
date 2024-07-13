from TTS.api import TTS
import gradio as gr
import requests

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
        """
    )
    with gr.Tab("Train VITTS"):
        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                run_name = gr.Textbox(label="Run Name", placeholder="vits_ljspeech")
                batch_size = gr.Slider(label="Batch Size", minimum=0, maximum=64, step=8, value=32, info="Choose batch size for training. Typically a power of 2.")
                eval_batch_size = gr.Slider(label="Eval Batch Size", minimum=0, maximum=64, step=8, value=32, info="Choose batch size for evaluation. Typically a power of 2.")
                batch_group_size = gr.Slider(label="Batch Group Size", minimum=0, maximum=64, step=8, value=32, info="Choose batch group size. Typically a power of 2.")
                num_loader_workers = gr.Slider(label="Num Loader Workers", minimum=1, maximum=20, step=1, value=4, info="Number of workers for data loading.")

            with gr.Column(scale=1, min_width=600):
                run_eval = gr.Checkbox(label="Run Eval", value=True, info="Whether to run evaluation after training.")
                num_eval_loader_workers = gr.Slider(label="Num Eval Loader Workers", minimum=1, maximum=20, step=1, value=4, info="Number of workers for evaluation data loading.")
                test_delay_epochs = gr.Number(label="Test Delay Epochs", value=-1, info="Number of epochs to delay evaluation. Set to -1 for no delay.")
                epochs = gr.Number(label="Epochs", value=25000, info="Total number of training epochs.")
                text_cleaner = gr.Textbox(label="Text Cleaner", value="english_cleaners", interactive=True, info="Text cleaner to use. Default is 'english_cleaners'.")

        with gr.Row():
            with gr.Column(scale=1, min_width=600):
                use_phonemes = gr.Checkbox(label="Use Phonemes", value=False, info="Whether to use phonemes.")
                phoneme_language = gr.Textbox(label="Phoneme Language", value="en-us", interactive=True, info="Language for phonemes. Default is 'en-us'.")
                phoneme_cache_path = gr.Textbox(label="Phoneme Cache Path", value="phoneme_cache", interactive=True, info="Path to cache phonemes.")

            with gr.Column(scale=1, min_width=600):
                compute_input_seq_cache = gr.Checkbox(label="Compute Input Seq Cache", value=False, info="Whether to compute input sequence cache.")
                print_step = gr.Number(label="Print Step", value=25, info="Step interval for printing logs.")
                print_eval = gr.Checkbox(label="Print Eval", value=False, info="Whether to print evaluation results.")
                mixed_precision = gr.Checkbox(label="Mixed Precision", value=False, info="Whether to use mixed precision training.")

        output_path = gr.Textbox(label="Output Path", value="/path/to/output", info="Directory path for outputs.")
        datasets = gr.Textbox(label="Datasets", value="metadata.csv", info="Path to dataset file.")
        test_sentences = gr.Textbox(label="Test Sentences", value="Sentence 1\nSentence 2\nSentence 3", info="Sentences for testing the model.")

        cudnn_benchmark = gr.Checkbox(label="CUDNN Benchmark", value=False, info="Enable CUDNN benchmark for faster training.")

        with gr.Row():
            button = gr.Button("Submit", variant="primary")

        output = gr.Textbox(label="Output")

        def run_voice_clonify(run_name, batch_size, eval_batch_size, batch_group_size, num_loader_workers, run_eval, 
                              num_eval_loader_workers, test_delay_epochs, epochs, text_cleaner, use_phonemes, phoneme_language, 
                              phoneme_cache_path, compute_input_seq_cache, print_step, print_eval, mixed_precision, 
                              output_path, datasets, cudnn_benchmark, test_sentences):
            config_params = {
                "run_name": run_name,
                "batch_size": batch_size,
                "eval_batch_size": eval_batch_size,
                "batch_group_size": batch_group_size,
                "num_loader_workers": num_loader_workers,
                "num_eval_loader_workers": num_eval_loader_workers,
                "run_eval": run_eval,
                "test_delay_epochs": test_delay_epochs,
                "epochs": epochs,
                "text_cleaner": text_cleaner,
                "use_phonemes": use_phonemes,
                "phoneme_language": phoneme_language,
                "phoneme_cache_path": phoneme_cache_path,
                "compute_input_seq_cache": compute_input_seq_cache,
                "print_step": print_step,
                "print_eval": print_eval,
                "mixed_precision": mixed_precision,
                "output_path": output_path,
                "datasets": datasets,
                "cudnn_benchmark": cudnn_benchmark,
                "test_sentences": test_sentences.split("\n"),
            }
            
            # Print the parameters to check the values
            print("Config parameters received:")
            for key, value in config_params.items():
                print(f"{key}: {value}")

            response = requests.post("http://localhost:5000/train", json=config_params)
            return response.json()

        inputs = [
            run_name, batch_size, eval_batch_size, batch_group_size, num_loader_workers, run_eval,
            num_eval_loader_workers, test_delay_epochs, epochs, text_cleaner, use_phonemes, phoneme_language,
            phoneme_cache_path, compute_input_seq_cache, print_step, print_eval, mixed_precision, output_path,
            datasets, cudnn_benchmark, test_sentences
        ]
        button.click(run_voice_clonify, inputs=inputs, outputs=[output])

demo.launch(share=True)
