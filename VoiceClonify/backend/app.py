from flask import Flask, request, jsonify
from multiprocessing import Process, freeze_support
import train_vits_script  # Import the modified backend script

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    print("Received request to start training")
    params = request.json
    print(f"Parameters received: {params}")
    p = Process(target=train_vits_script.train_vits, args=(params,))
    p.start()
    print("Training process started")
    return jsonify({"status": "training started"}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    freeze_support()
    app.run(host='0.0.0.0', port=5000)
    print("Flask server running")
