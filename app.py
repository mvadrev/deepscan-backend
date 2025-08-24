from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
import numpy as np

# -------------------------------
# Load YOLO model ONCE at startup
# -------------------------------
try:
    model = YOLO("./best.pt")  # replace with your trained weights
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print(f"❌ Failed to load YOLO model: {e}")
    model = None

app = Flask(__name__)

# -------------------------------
# Hello / Health-check route
# -------------------------------
@app.route("/", methods=["GET"])
def home():
    return "Server running ✅", 200

@app.route("/upload", methods=["POST"])
def upload():
    if model is None:
        return jsonify({"error": "Model not loaded on server"}), 500

    # -------------------------------
    # Validate input
    # -------------------------------
    img_data = request.data
    if not img_data:
        return jsonify({"error": "No data received"}), 400

    # Convert raw bytes to image
    try:
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    except Exception:
        return jsonify({"error": "Failed to decode image"}), 400

    if img is None:
        return jsonify({"error": "Invalid image data"}), 400

    # Resize to reduce memory usage
    img = cv2.resize(img, (640, 640))

    # -------------------------------
    # Run YOLO inference
    # -------------------------------
    try:
        results = model.predict(img, verbose=False)
    except Exception as e:
        return jsonify({"error": f"Inference failed: {str(e)}"}), 500

    # -------------------------------
    # Count pills
    # -------------------------------
    count = sum(len(r.boxes) for r in results)
    print(f"📦 Pills detected: {count}")

    return jsonify({
        "pill_count": int(count),
        "detections": [
            {
                "class": int(box.cls[0]),
                "confidence": float(box.conf[0])
            }
            for r in results
            for box in r.boxes
        ]
    }), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
