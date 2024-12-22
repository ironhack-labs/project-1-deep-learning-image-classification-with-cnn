import os
import json
import numpy as np
from flask import Flask, request, render_template, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image
from google.auth import default
from google.auth.transport.requests import Request
import requests
from PIL import Image

# Flask app setup
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['PROCESSED_FOLDER'] = 'processed'

# Ensure the upload and processed directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['PROCESSED_FOLDER'], exist_ok=True)

# Function to preprocess the image
def preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  # (32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension -> (1, 32, 32, 3)
    img_array /= 255.0  # Normalize to [0, 1] range

    # Save the preprocessed image for visualization
    processed_img_path = os.path.join(app.config['PROCESSED_FOLDER'], os.path.basename(img_path))
    img_resized = Image.fromarray((img_array[0] * 255).astype(np.uint8))
    img_resized.save(processed_img_path)

    return img_array, processed_img_path

# Function to send prediction request
def predict_image(img_path):
    # Preprocess the image
    img_array, processed_img_path = preprocess_image(img_path)
    input_data = img_array.tolist()

    # Prepare the payload
    data = json.dumps({"instances": input_data})
    url = 'https://us-east1-aiplatform.googleapis.com/v1/projects/10609508497/locations/us-east1/endpoints/2868357556030406656:predict'

    # Authenticate using service account credentials
    credentials, project = default()
    credentials.refresh(Request())

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {credentials.token}"
    }

    # Send the request
    response = requests.post(url, headers=headers, data=data)
    response_data = response.json()

    # Parse predictions
    if 'predictions' in response_data:
        predictions = response_data['predictions'][0]
        return predictions, processed_img_path
    else:
        return {"error": response_data.get("error", "Unknown error")}, processed_img_path

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    images = request.files.getlist('images')
    results = []

    for img in images:
        filename = secure_filename(img.filename)
        img_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        img.save(img_path)
        predictions, processed_img_path = predict_image(img_path)

        if isinstance(predictions, dict) and "error" in predictions:
            results.append({"filename": filename, "error": predictions["error"]})
        else:
            probabilities = {class_names[i]: float(prob) for i, prob in enumerate(predictions)}
            predicted_class = class_names[np.argmax(predictions)]
            results.append({
                "filename": filename,
                "processed_image": os.path.basename(processed_img_path),  # Use basename here
                "predicted_class": predicted_class,
                "probabilities": probabilities
            })

    return render_template('results.html', results=results)


@app.route('/processed/<filename>')
def processed_image(filename):
    return send_from_directory(app.config['PROCESSED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)
