import gradio as gr
import requests
import os
import shutil
from datetime import datetime
import bcrypt

theme = gr.themes.Default(
    primary_hue="blue",
    secondary_hue="slate",
    font=[gr.themes.GoogleFont('Inter')],
    font_mono=[gr.themes.GoogleFont('Fira Mono')],
)

USER_CREDENTIALS_FILE = "./user_credentials.txt"

def hash_password(password):
    return bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())

def check_password(stored_password, provided_password):
    return bcrypt.checkpw(provided_password.encode('utf-8'), stored_password.encode('utf-8'))

def register_user(username, password):
    hashed_password = hash_password(password).decode('utf-8')
    with open(USER_CREDENTIALS_FILE, 'a') as file:
        file.write(f"{username},{hashed_password}\n")

def check_user_credentials(username, password):
    if os.path.exists(USER_CREDENTIALS_FILE):
        with open(USER_CREDENTIALS_FILE, 'r') as file:
            credentials = file.read().splitlines()
            for credential in credentials:
                stored_username, stored_password = credential.split(',')
                if stored_username == username and check_password(stored_password, password):
                    return True
    return False

def username_exists(username):
    if os.path.exists(USER_CREDENTIALS_FILE):
        with open(USER_CREDENTIALS_FILE, 'r') as file:
            credentials = file.read().splitlines()
            for credential in credentials:
                stored_username, _ = credential.split(',')
                if stored_username == username:
                    return True
    return False

def handle_credentials(username, password, action):
    if action == "Register":
        if username_exists(username):
            return "Username already exists. Please choose a different username.", False, action
        register_user(username, password)
        return f"User '{username}' registered successfully.", False, action
    elif action == "Sign In":
        if check_user_credentials(username, password):
            return f"User '{username}' signed in successfully.", True, action
        else:
            return "Invalid username or password.", False, action

def upload_dataset(files, username, password):
    if not check_user_credentials(username, password):
        return "", "Invalid username or password."
    
    base_folder = "../datasets"
    # Create a new folder for the current upload
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    user_folder = os.path.join(base_folder, f"{username}_{timestamp}")
    os.makedirs(user_folder, exist_ok=True)
    
    # Copy the uploaded files to the new folder
    for file in files:
        shutil.copyfile(file.name, os.path.join(user_folder, os.path.basename(file.name)))
    
    return user_folder, f"Dataset uploaded successfully to {user_folder}!"

def run_voice_clonify(run_name, batch_size, eval_batch_size, batch_group_size, num_loader_workers, run_eval, 
                      num_eval_loader_workers, test_delay_epochs, epochs, text_cleaner, use_phonemes, phoneme_language, 
                      compute_input_seq_cache, print_step, print_eval, mixed_precision, 
                      dataset_path, cudnn_benchmark, test_sentences):
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
        "compute_input_seq_cache": compute_input_seq_cache,
        "print_step": print_step,
        "print_eval": print_eval,
        "mixed_precision": mixed_precision,
        "datasets": dataset_path,
        "cudnn_benchmark": cudnn_benchmark,
        "test_sentences": test_sentences.split("\n"),
    }

    print("TEST")
    print(config_params)

    response = requests.post("http://localhost:5000/train", json=config_params)
    if response.status_code == 200:
        return "Training started successfully!"
    else:
        return f"Training failed with status code: {response.status_code}\n{response.text}"

with gr.Blocks(theme=theme, title="Voice Clonify") as demo:
    gr.Markdown(
        """
        # Voice Clonify
        Clone your very own voices for text to speech generation 🗣️.
        """
    )

    login_state = gr.State(value=False)
    action_state = gr.State(value="")

    with gr.Tab("Credentials"):
        gr.Markdown(
            """
            # Submit Credentials
            Register or sign in with your username and password.
            """
        )
        username = gr.Textbox(label="Username", placeholder="Enter your username")
        password = gr.Textbox(label="Password", type="password", placeholder="Enter your password")
        action = gr.Radio(choices=["Register", "Sign In"], label="Action", value="Sign In")
        submit_credentials_button = gr.Button(value="Submit", variant="primary")
        credentials_status = gr.Textbox(label="Status", interactive=False)
        submit_credentials_button.click(handle_credentials, inputs=[username, password, action], outputs=[credentials_status, login_state, action_state])

    with gr.Tab("Upload Dataset") as upload_tab:
        upload_column = gr.Column(visible=False)
        with upload_column:
            gr.Markdown(
                """
                # Upload Dataset
                Upload your dataset for training.
                """
            )
            upload_files = gr.File(file_count="directory")
            upload = gr.Button(value="Upload", variant="primary")
            dataset_path = gr.Textbox(label="Dataset Path", interactive=False)
            upload_status = gr.Textbox(label="Upload Status", interactive=False)
            upload.click(upload_dataset, inputs=[upload_files, username, password], outputs=[dataset_path, upload_status])

    with gr.Tab("Train VITTS") as train_tab:
        train_column = gr.Column(visible=False)
        with train_column:
            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    run_name = gr.Textbox(label="Run Name", placeholder="vits_ljspeech")
                    batch_size = gr.Slider(label="Batch Size", minimum=0, maximum=64, step=8, value=32, info="Choose batch size for training.")
                    eval_batch_size = gr.Slider(label="Eval Batch Size", minimum=0, maximum=64, step=8, value=16, info="Choose batch size for evaluation.")
                    batch_group_size = gr.Slider(label="Batch Group Size", minimum=0, maximum=5, step=1, value=5, info="Choose batch group size.")
                    num_loader_workers = gr.Slider(label="Num Loader Workers", minimum=1, maximum=20, step=1, value=4, info="Number of workers for loading training data.")

                with gr.Column(scale=1, min_width=600):
                    run_eval = gr.Checkbox(label="Run Eval", value=True, info="Whether to run evaluation after training.")
                    num_eval_loader_workers = gr.Slider(label="Num Eval Loader Workers", minimum=1, maximum=20, step=1, value=4, info="Number of workers for loading evaluation data.")
                    test_delay_epochs = gr.Number(label="Test Delay Epochs", value=-1, info="Number of epochs to delay evaluation. Set to -1 for no delay.")
                    epochs = gr.Number(label="Epochs", value=25000, info="Total number of training epochs.")
                    text_cleaner = gr.Textbox(label="Text Cleaner", value="english_cleaners", interactive=True, info="Text cleaner to use. Default is 'english_cleaners'.")

            with gr.Row():
                with gr.Column(scale=1, min_width=600):
                    use_phonemes = gr.Checkbox(label="Use Phonemes", value=False, info="Whether to use phonemes.")
                    phoneme_language = gr.Textbox(label="Phoneme Language", value="en-us", interactive=True, info="Language for phonemes. Default is 'en-us'.")

                with gr.Column(scale=1, min_width=600):
                    compute_input_seq_cache = gr.Checkbox(label="Compute Input Seq Cache", value=False, info="Whether to compute input sequence cache.")
                    print_step = gr.Number(label="Print Step", value=25, info="Step interval for printing logs.")
                    print_eval = gr.Checkbox(label="Print Eval", value=False, info="Whether to print evaluation results.")
                    mixed_precision = gr.Checkbox(label="Mixed Precision", value=False, info="Whether to use mixed precision training.")

            test_sentences = gr.Textbox(label="Test Sentences", value="Sentence 1\nSentence 2\nSentence 3", info="Sentences for testing the model.")
            cudnn_benchmark = gr.Checkbox(label="CUDNN Benchmark", value=False, info="Enable CUDNN benchmark for faster training.")

            with gr.Row():
                button = gr.Button("Submit", variant="primary")
            output = gr.Textbox(label="Output")

            inputs = [
                run_name, batch_size, eval_batch_size, batch_group_size, num_loader_workers, run_eval,
                num_eval_loader_workers, test_delay_epochs, epochs, text_cleaner, use_phonemes, phoneme_language,
                compute_input_seq_cache, print_step, print_eval, mixed_precision,
                dataset_path, cudnn_benchmark, test_sentences
            ]

            outputs = [output]
            button.click(run_voice_clonify, inputs=inputs, outputs=outputs)

    def toggle_visibility(login_status, action_status):
        if action_status == "Sign In" and login_status:
            return [gr.update(visible=True), gr.update(visible=True)]
        return [gr.update(visible=False), gr.update(visible=False)]

    login_state.change(toggle_visibility, inputs=[login_state, action_state], outputs=[upload_column, train_column])

demo.launch(share=True)
