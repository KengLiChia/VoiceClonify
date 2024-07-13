import gradio as gr
import requests
from urllib.parse import urljoin

API_ENDPOINT = "http://127.0.0.1:5000/"

def upload_prompt(file):
    try:
        full_url = urljoin(API_ENDPOINT, "api/upload_prompt")
        
        with open(file.name, 'rb') as f:
            files = {'file': (file.name, f, 'text/csv')}
            print(file.name)
            response = requests.post(full_url, files=files)
        if response.status_code == 200:
            return file.name
        else:
            return f"Failed to upload file. Status code: {response.status_code}"
    except Exception as e:
        return f"Error uploading file: {str(e)}"

with gr.Blocks() as demo:
    gr.Markdown("# Upload CSV File to API Demo")
    file_output = gr.File()
    upload_button = gr.UploadButton(label="Upload CSV File", file_count="single", file_types=[".csv"])
    upload_button.upload(upload_prompt, upload_button, file_output)

if __name__ == "__main__":
    demo.launch(share=True)
