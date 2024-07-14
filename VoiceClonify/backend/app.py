from flask import Flask, request, jsonify
from multiprocessing import Process
import train_vits_script

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    params = request.json
    print(f"Parameters received: {params}")
    process = Process(target=train_vits_script.train_vits, args=(params,))
    process.start()
    print("Training process started")
    return jsonify({"status": "training started"}), 200

if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host='0.0.0.0', port=5000)
    print("Flask server running")
