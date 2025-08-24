from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

# Load your YOLO model (adjust path if needed)
model = YOLO("./best.pt")  # replace with your trained model path

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    # Get raw JPEG bytes from ESP32
    img_data = request.data
    if not img_data:
        return jsonify({"error": "No data received"}), 400

    # Convert bytes to numpy array
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Run YOLO inference
    results = model.predict(img, verbose=False)  # verbose=False to suppress print

    # Count number of pills detected
    count = 0
    for result in results:
        count += len(result.boxes)  # each box is a detected object

    print(f"Pills detected: {count}")
    return jsonify({"pill_count": count}), 200

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
