from flask import Flask, request, render_template, redirect, url_for
from PIL import Image
from tensorflow.keras.models import load_model
import numpy as np
import os
import logging
import traceback

app = Flask(__name__)

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load models at startup
models = {}
try:
    model_dir = 'saved_models'
    if not os.path.exists(model_dir):
        raise Exception(f"Directory {model_dir} not found")
    
    for model_file in os.listdir(model_dir):
        if model_file.endswith('.keras'):
            model_path = os.path.join(model_dir, model_file)
            models[model_file] = load_model(model_path)
            logger.info(f"Loaded model: {model_file}")
except Exception as e:
    logger.error(f"Error loading models: {str(e)}")
    logger.error(traceback.format_exc())

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return redirect(url_for('home'))
        
    try:
        if 'image' not in request.files:
            raise ValueError("No image file provided")
        
        file = request.files['image']
        if file.filename == '':
            raise ValueError("No selected file")

        img = Image.open(file)
        img = img.convert('RGB')  # Ensure image is in RGB format
        img = img.resize((32, 32))
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        results = []
        for model_name, model in models.items():
            pred = model.predict(img_array)
            class_idx = np.argmax(pred[0])
            confidence = float(pred[0][class_idx])
            results.append({
                'model': model_name,
                'class': int(class_idx),
                'confidence': f"{confidence:.4f}"
            })

        return render_template('index.html', results=results)

    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    app.run(port=5001, debug=True)