from flask import Flask, request, render_template
import tensorflow as tf
import numpy as np
from PIL import Image

# Initialize the Flask app (must be before defining routes)
app = Flask(__name__)

# Load your pre-trained model (make sure to specify the correct path)
model = tf.keras.models.load_model("model.h5")  # Adjust to your model's path

# Class labels (for example, CIFAR-10)
labels = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Define a route for the homepage
@app.route('/')
def home():
    return render_template('upload.html')  # Render the upload.html template

# Define a route for handling image uploads and predictions
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded. Please select a file."

    file = request.files['file']

    if file.filename == '':
        return "No file selected. Please select a valid image file."

    if file:
        try:
            # Open the image and convert to RGB to ensure it has 3 channels
            img = Image.open(file).convert('RGB')
            
            # Resize to match the model's input size
            img = img.resize((32, 32))
            
            # Convert the image to a NumPy array and normalize it
            img_array = np.array(img)
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
            img_array = img_array / 255.0  # Normalize the image

            # Make predictions
            predictions = model.predict(img_array)
            predicted_class = np.argmax(predictions)
            predicted_label = labels[predicted_class]

            # Get probabilities for each class
            probabilities = predictions[0]
            class_probabilities = {label: f"{prob:.2f}%" for label, prob in zip(labels, probabilities * 100)}

            # Return the predicted class and probabilities
            return render_template('result.html', predicted_label=predicted_label, class_probabilities=class_probabilities)

        except Exception as e:
            return f"Error processing the image: {str(e)}"
    
    return "Something went wrong. Please try again."

# Run the app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
