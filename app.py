import os
import numpy as np
from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
import threading

# ----------------------------
# Import TFLite interpreter
# ----------------------------
try:
    import tflite_runtime.interpreter as tflite  # Heroku Linux
except ImportError:
    import tensorflow as tf
    tflite = tf.lite

# ----------------------------
# Flask app
# ----------------------------
app = Flask(__name__)

# Folder to temporarily save uploaded images
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# ----------------------------
# Model path and class labels
# ----------------------------
MODEL_PATH = "tomato_leaf_model.tflite"

CLASS_NAMES = [
    "Bacterial_spot",
    "Early_blight",
    "Late_blight",
    "Leaf_Mold",
    "Septoria_leaf_spot",
    "Spider_mites",
    "Target_Spot",
    "Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato_mosaic_virus",
    "Healthy"
]

# ----------------------------
# Load TFLite model once (thread-safe)
# ----------------------------
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
lock = threading.Lock()

# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("home.html", result="No image uploaded")

    file = request.files["file"]
    if file.filename == "":
        return render_template("home.html", result="No image selected")

    # Save uploaded image
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    # Preprocess image
    img = Image.open(filepath).convert("RGB")
    img = img.resize((224, 224))
    img_array = np.array(img).astype(np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # TFLite prediction (thread-safe)
    with lock:
        interpreter.set_tensor(input_details[0]["index"], img_array)
        interpreter.invoke()
        predictions = interpreter.get_tensor(output_details[0]["index"])

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions)) * 100

    return render_template(
        "home.html",
        result=f"Disease: {predicted_class}",
        confidence=f"Confidence: {confidence:.2f}%",
        img_path=filepath
    )


# ----------------------------
# Run app
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)


