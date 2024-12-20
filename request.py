from tensorflow.keras.preprocessing import image
import numpy as np
import requests
import json

# Load and preprocess the image
def preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

# Path to the image you want to predict
img_path = './image-prediction/images/360_F_177742846_umwpEr5OqwEQd4a9VyS7BGJX3tINNDe7.jpg'  

# Preprocess the image
img_array = preprocess_image(img_path)

# Prepare the request
input_data = np.expand_dims(img_array, axis=0).tolist()

# Send the request to TensorFlow Serving
data = json.dumps({"instances": input_data})
headers = {"content-type": "application/json"}
response = requests.post('http://localhost:8501/v1/models/my_model:predict', data=data, headers=headers)

# Print the predicted class
predictions = response.json()

# Class names
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

prediction_values = predictions['predictions'][0]

# Get the index of the class with the highest probability
predicted_class_index = np.argmax(prediction_values)

# Get the class name corresponding to that index
predicted_class_name = class_names[predicted_class_index]

# Print the predicted class name
print(f"Predicted Class: {predicted_class_name}")

