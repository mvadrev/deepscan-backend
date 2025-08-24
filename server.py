from flask import Flask, request

app = Flask(__name__)

@app.route("/upload", methods=["POST"])
def upload():
    # ESP32 sends raw JPEG bytes
    img_data = request.data
    if not img_data:
        return "No data received", 400

    # Save the image
    with open("upload.jpg", "wb") as f:
        f.write(img_data)

    print("Image saved: upload.jpg")
    return "Image saved", 200

if __name__ == "__main__":
    # Run on all interfaces
    app.run(host="0.0.0.0", port=5000)
