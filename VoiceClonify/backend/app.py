from flask import Flask, request, jsonify, send_file
from multiprocessing import Process
import train_vits_script
import os
import glob

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

@app.route('/logs', methods=['GET'])
def get_logs():
    output_path = request.args.get('path')
    # Use glob to get the latest directory
    list_of_files = glob.glob(os.path.join(output_path, '*/training_log.txt'))
    if not list_of_files:
        return jsonify({"error": "No log files found"}), 404
    
    latest_file = max(list_of_files, key=os.path.getctime)
    if os.path.exists(latest_file):
        return send_file(latest_file)
    else:
        return jsonify({"error": "Log file not found"}), 404
    
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
    print("Flask server running")
