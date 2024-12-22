import requests
import json
import numpy as np
from tensorflow.keras.preprocessing import image
from google.auth import default
from google.auth.transport.requests import Request

# Load and preprocess the image
def preprocess_image(img_path, target_size=(32, 32)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)  # (32, 32, 3)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension -> (1, 32, 32, 3)
    img_array /= 255.0  # Normalize to [0, 1] range
    return img_array

# Load the image
img_path = 'image-prediction/images/360_F_177742846_umwpEr5OqwEQd4a9VyS7BGJX3tINNDe7.jpg'
img_array = preprocess_image(img_path)

# Debug input shape
print("Input shape:", img_array.shape)  # Should print (1, 32, 32, 3)

# Convert to JSON serializable format
input_data = img_array.tolist()

# Prepare the payload
data = json.dumps({"instances": input_data})

# Define your endpoint URL
url = 'https://us-east1-aiplatform.googleapis.com/v1/projects/10609508497/locations/us-east1/endpoints/2868357556030406656:predict'

# Authenticate using service account credentials
credentials, project = default()
credentials.refresh(Request())

# Add the authorization header
headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {credentials.token}"
}

# Send the request
response = requests.post(url, headers=headers, data=data)

# Print the entire response for debugging
response_data = response.json()
print("Response:", response_data)

# Check for predictions
if 'predictions' in response_data:
    predictions = response_data['predictions']
    prediction_values = predictions[0]  # Assuming there's one result
    predicted_class_index = np.argmax(prediction_values)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    print(f"Predicted class: {class_names[predicted_class_index]}")
else:
    print("Error: 'predictions' not found in the response.")
