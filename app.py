import os
import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.preprocessing import image

# Prevent TensorFlow crash
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

st.title("Group 4: Neural Network Image Classification ")
st.write("This project showcases an end-to-end deep learning workflow for image classification on the CIFAR-10 dataset. Group 4 designed, trained, and evaluated multiple convolutional neural network models—including a custom baseline CNN and transfer learning with VGG16.")

classnames = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

# Preprocessing function - ensures uploaded images match training images
def preprocess_uploaded_image(uploaded_file):
    img = image.load_img(uploaded_file, target_size=(32, 32))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # normalize like during training
    img_array = np.expand_dims(img_array, axis=0)  # shape: (1, 32, 32, 3)
    return img_array

@st.cache_resource
def load_model_safely(path):
    from tensorflow import keras
    return keras.models.load_model(path, compile=False)

# Model options
model_paths = {
    "Roland Baseline": "best_baseline.h5",
    "Roland VGG16_Transfer": "models/vgg16_transfer.h5",
    "Ralitza Model": "ralitza_m1.h5",
    "Ishu CNN": "ishu_cnn_cifar10_model.h5"
}



model_choice = st.selectbox("Select Model", options=list(model_paths.keys()))

try:
    model = load_model_safely(model_paths[model_choice])
    st.success(f"{model_choice} model loaded.")
except Exception as e:
    st.error(f"Could not load {model_choice} model: {e}")
    st.stop()

uploaded_file = st.file_uploader("Upload an image:", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", width=200)

    # Preprocess the uploaded image properly
    preprocessed_img = preprocess_uploaded_image(uploaded_file)

    # Make prediction
    preds = model.predict(preprocessed_img, verbose=0)
    pred_class = classnames[np.argmax(preds)]

    st.markdown(f"**Selected Model:** {model_choice}")
    st.markdown(f"**Prediction:** {pred_class}")
