import os
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Initialize Flask app
app = Flask(__name__)

# Load trained CNN model (load once)
MODEL_PATH = "cnn_model.h5"
model = load_model(MODEL_PATH)

# Class labels (MUST match training order)
CLASS_NAMES = [
    'Bacterial_spot',
    'Early_blight',
    'Late_blight',
    'Leaf_Mold',
    'Septoria_leaf_spot',
    'Spider_mites',
    'Target_Spot',
    'Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato_mosaic_virus',
    'Healthy'
]

# Home route
@app.route('/')
def home():
    return render_template('home.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', result="No file uploaded")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', result="No file selected")

    # Save image temporarily
    img_path = os.path.join("static", file.filename)
    file.save(img_path)

    # Load and preprocess image
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    return render_template(
        'index.html',
        result=f"Disease: {predicted_class}",
        confidence=f"Confidence: {confidence:.2f}%"
    )

# Run app
if __name__ == "__main__":
    app.run(debug=True)
