"""Step 4: a small web app. Upload pictures and the CNN says what it sees."""
from pathlib import Path

import keras
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from project_tools import CLASS_NAMES

MODEL_FILE = Path(__file__).parent / "models" / "cnn_cifar10.keras"


@st.cache_resource
def get_model():
    """Load the trained CNN once and keep it in memory."""
    return keras.models.load_model(MODEL_FILE)


def prepare_picture(uploaded_file):
    """Turn an uploaded picture into the 32 x 32 format the CNN was trained on."""
    picture = Image.open(uploaded_file).convert("RGB")
    small = picture.resize((32, 32))
    pixels = np.asarray(small).astype("float32") / 255
    return picture, pixels.reshape(1, 32, 32, 3)


st.title("CIFAR-10 picture classifier")
st.write("Upload one or more pictures. My CNN says which of the 10 classes it sees, "
         "with the probability for every class.")
st.caption("Classes: " + ", ".join(CLASS_NAMES))

uploaded_files = st.file_uploader("Choose pictures", type=["png", "jpg", "jpeg"],
                                  accept_multiple_files=True)

model = get_model()

for uploaded_file in uploaded_files:
    picture, pixels = prepare_picture(uploaded_file)
    probabilities = model.predict(pixels, verbose=0)[0]
    best = int(np.argmax(probabilities))
    best_percent = round(float(probabilities[best]) * 100, 1)

    table = pd.DataFrame({"class": CLASS_NAMES,
                          "probability (%)": (probabilities * 100).round(1)})
    table = table.sort_values("probability (%)", ascending=False)

    left, right = st.columns([1, 2])
    with left:
        st.image(picture, caption=uploaded_file.name, width=200)
    with right:
        st.subheader(f"{CLASS_NAMES[best]} ({best_percent} %)")
        st.dataframe(table, hide_index=True)
    st.divider()