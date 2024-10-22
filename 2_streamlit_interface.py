# %%
import streamlit as st
from PIL import Image
import numpy as np
import pickle
from tensorflow import keras

# %%
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

label_names = unpickle("data/batches.meta")

# %%
st.set_page_config(layout="wide", page_title="Image Classification")

st.write("## Classify your image in one of 10 classes")
st.sidebar.write("## Upload and download")

MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

def classify_image(upload):
    image = Image.open(upload)
    model = pickle.load(open("trained_model_whole_dataset.pkl", "rb"))
    # image = np.pad(image, ((0, 0), (0, 0), (0, 3072 - (image.size[0]*image.size[1]) % 3072)), mode='constant')
    # image = np.asarray(image).reshape(None, 3, 32, 32).transpose(0,2,3,1)
    label = model.predict(np.expand_dims(np.asarray(image),axis=0))
    return np.argmax(label)

col1, col2 = st.columns(2)
col1.write("Original Image")
col2.write("Label")
my_upload = st.sidebar.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if my_upload is not None:
    if my_upload.size > MAX_FILE_SIZE:
        st.error("The uploaded file is too large. Please upload an image smaller than 5MB.")
    else:
        col1.image(my_upload, width=500)
        label = classify_image(upload=my_upload)
        col2.write(label_names[b'label_names'][label])
else:
    pass


