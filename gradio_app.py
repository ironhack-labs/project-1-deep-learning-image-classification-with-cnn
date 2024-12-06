import gradio as gr
import tensorflow as tf
import numpy as np

#load model
model = tf.keras.models.load_model('model_nameg.keras')

# class names
class_names = ["Airplane", "Cute automobile", "Cute Bird", "Kitty", "Deer", 
               "Dog", "Cute frog", "Horse", "Ship", "Truck"]

#preprocess image and prediction
def predict_image(image):
    image_resized = tf.image.resize(image, (32, 32))
    mean = image.mean(axis=(0,1,2), keepdims=True)
    std = image.std(axis=(0,1,2), keepdims=True)
    image_normalized = (image_resized - mean) / std
    image_batch = np.expand_dims(image_normalized, axis=0)
    
    predictions = model.predict(image_batch)
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])

    return f"Class: {predicted_class} (Confidence: {confidence:.2f})"

#interface
interface = gr.Interface(
    fn=predict_image,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=gr.Textbox(label="Prediction"),
    title="Image Classifier",
    description="Upload an image to get a prediction."
)

# Launch
interface.launch(share = True)
