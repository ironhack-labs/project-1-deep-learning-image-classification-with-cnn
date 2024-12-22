# Image Classification Web App

## Overview
This web application allows users to classify images into predefined categories using a Convolutional Neural Network (CNN). Users can upload one or multiple images, view their preprocessed versions, and receive predictions with associated probabilities. The application integrates a custom-trained deep learning model hosted on Google Cloud's Vertex AI platform.

## Features
- **Image Upload**: Users can upload one or multiple images through a web interface.
- **Predictions**: The app displays the predicted class labels and associated probabilities.
- **Preprocessed Image Display**: Shows the preprocessed version of uploaded images to illustrate how the model interprets input data.

## Technology Stack
### Frontend
- **HTML**: Provides the structure for web pages.
- **HTMX**: Enables dynamic updates without full-page reloads.
- **Tailwind CSS**: Ensures a clean and professional design.

### Backend
- **Flask (Python)**: Handles image uploads, preprocessing, and API requests.

### Model API
- **Google AI Platform**: Hosts the trained CNN model and provides an endpoint for making predictions.

## Workflow
1. **User Interaction**:
    - Users visit the web application and upload images.
2. **Image Preprocessing**:
    - Images are resized to 32x32 pixels and normalized to match the model's input requirements.
3. **Model Prediction**:
    - Preprocessed images are sent to the Google AI Platform endpoint for prediction.
    - Predictions include class labels and their probabilities.
4. **Results Display**:
    - The app displays predictions and preprocessed images.

## Installation and Setup

### Prerequisites
- Python 3.11 or later
- Google Cloud SDK (configured with service account credentials)
- Docker (optional, for local TensorFlow Serving)

### Steps
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd image-classification-webapp
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Set Environment Variables**:
   Configure Google Cloud credentials:
   ```bash
   export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
   ```

4. **Run the Application**:
   ```bash
   flask run
   ```
   The application will be accessible at `http://127.0.0.1:5000`.

## Usage
1. Navigate to the homepage.
2. Upload one or multiple images using the provided form.
3. View predictions and preprocessed images on the results page.

## Deployment
The application can be deployed to any server or cloud platform that supports Flask. For production use:
- Use a WSGI server like Gunicorn.
- Configure HTTPS.

## Folder Structure
```
image-classification-webapp/
├── app.py                # Main application script
├── static/               # Static files (CSS, JS, images)
├── templates/            # HTML templates
├── requirements.txt      # Python dependencies
└── README.md             # Project documentation
```

## Acknowledgments
- **TensorFlow**: For providing tools to preprocess data and train models.
- **Google Cloud Vertex AI**: For hosting the model.
- **Flask**: For powering the web application backend.
- **Tailwind CSS**: For the frontend design.

## License
This project is licensed under the MIT License.

