# Image Classifier

This is a simple image classification application using Flask for the backend and a Tailwind CSS-based UI. It allows users to upload an image and get predictions from a pre-trained TensorFlow model.

---

## Features
- Upload images via a user-friendly web interface.
- Classify images using a pre-trained TensorFlow model.
- Display prediction probabilities for each class.

---

## Requirements
- Python 3.8, 3.9, or 3.10
- Pip (Python package installer)

---

## Installation and Setup Instructions

1. **Install Python Dependencies**
    Run the following command to install all required Python libraries:
   - pip install flask tensorflow pillow

---

2. **Download the Pre-trained Model**
    Ensure your TensorFlow model file (`Final.keras`) is located at the following path:
    - model\Final.keras

---

3. **Open the Application**
    Open your browser and navigate to:
    - http://localhost:5000

---

## Usage

1. Select an image file by clicking the "Choose File" button in the UI.
2. Click "Upload and Classify" to send the image to the server.
3. View the predicted class and probabilities for all classes in the UI.

---

### Error: TensorFlow Compatibility
Ensure you are using Python 3.8, 3.9, or 3.10. TensorFlow may not work with other Python versions.

