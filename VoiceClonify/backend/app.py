
from flask import Flask, request, jsonify
from multiprocessing import Process, freeze_support, Queue
import train_vits_script  # Import the modified backend script
import subprocess
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = '../model_uploads'
OUTPUT_FOLDER = '../generated_audios'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

@app.route('/train', methods=['POST'])
def train():
    print("Received request to start training")
    params = request.json
    print(f"Parameters received: {params}")

    queue = Queue() # Pass data between main process (Flask) and child process (training script)
    p = Process(target=train_vits_script.train_vits, args=(params, queue))
    p.start()
    p.join()  # Wait for the process to finish

    output_path = queue.get()
    print("Training process completed")
    return jsonify({"status": "training completed", "output_path": output_path}), 200


@app.route('/synthesize', methods=['POST'])
def synthesize():
    config_file = request.files['config_file']
    model_file = request.files['model_file']
    text = request.form['text']

    config_path = os.path.join(UPLOAD_FOLDER, secure_filename(config_file.filename))
    model_path = os.path.join(UPLOAD_FOLDER, secure_filename(model_file.filename))
    
    config_file.save(config_path)
    model_file.save(model_path)

    output_filename = f"generated.wav"
    output_path = os.path.join(OUTPUT_FOLDER, output_filename)

    command = f'python C:\\Users\\krist\\Documents\\GitHub\\VoiceClonify\\TTS-0.20.3\\TTS\\bin\\synthesize.py --config_path "{config_path}" --model_path "{model_path}" --text "{text}" --out_path "{output_path}"'
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode == 0:
        return jsonify({"status": "success", "output_path": output_path})
    else:
        return jsonify({"status": "error", "error_message": result.stderr}), 500

if __name__ == '__main__':
    print("Starting Flask server...")
    freeze_support()
    app.run(host='0.0.0.0', port=5000)
    print("Flask server running")