from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image

app = Flask(__name__, static_folder='.', static_url_path='')
CORS(app)  # Habilitar CORS para todas las rutas

# Cargar el modelo
model = tf.keras.models.load_model(r"model\Final.keras")
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def preprocess_image(image):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

@app.route('/')
def index():
    return send_from_directory('.', 'index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    try:
        image = Image.open(file.stream)
        processed_image = preprocess_image(image)
        predictions = model.predict(processed_image)
        predicted_class = class_names[np.argmax(predictions[0])]
        probabilities = {class_names[i]: float(predictions[0][i]) for i in range(10)}
        return jsonify({'predicted_class': predicted_class, 'probabilities': probabilities})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
