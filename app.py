import numpy as np
from flask import Flask, request, render_template
from PIL import Image
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    import tensorflow as tf
    tflite = tf.lite


# ----------------------------
# Flask App Initialization
# ----------------------------
app = Flask(__name__)

# ----------------------------
# Load TFLite Model (ONCE)
# ----------------------------
MODEL_PATH = "tomato_leaf_model.tflite"

interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ----------------------------
# Class Labels
# (MUST match training order)
# ----------------------------
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
# Routes
# ----------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return render_template("index.html", result="No image uploaded")

    file = request.files["file"]

    if file.filename == "":
        return render_template("index.html", result="No image selected")

    # ----------------------------
    # Image Preprocessing
    # ----------------------------
    img = Image.open(file).convert("RGB")
    img = img.resize((224, 224))

    img_array = np.array(img, dtype=np.float32)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # ----------------------------
    # Model Prediction
    # ----------------------------
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()

    predictions = interpreter.get_tensor(output_details[0]["index"])

    predicted_index = int(np.argmax(predictions))
    predicted_class = CLASS_NAMES[predicted_index]
    confidence = float(np.max(predictions)) * 100

    return render_template(
        "index.html",
        result=f"Disease: {predicted_class}",
        confidence=f"Confidence: {confidence:.2f}%"
    )

# ----------------------------
# Run App
# ----------------------------
if __name__ == "__main__":
    app.run(debug=True)

