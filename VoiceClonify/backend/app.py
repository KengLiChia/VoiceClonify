
from flask import Flask, request, jsonify
from multiprocessing import Process, freeze_support, Queue
import train_vits_script  # Import the modified backend script

app = Flask(__name__)

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

if __name__ == '__main__':
    print("Starting Flask server...")
    freeze_support()
    app.run(host='0.0.0.0', port=5000)
    print("Flask server running")
