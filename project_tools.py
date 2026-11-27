"""Shared tools for Project 1: the class names and loading CIFAR-10."""
from keras.applications.vgg16 import preprocess_input  # <-- NEW (line 2)
from keras.datasets import cifar10
from keras.utils import to_categorical

CLASS_NAMES = ["airplane", "automobile", "bird", "cat", "deer",
               "dog", "frog", "horse", "ship", "truck"]


def load_data():
    """Load CIFAR-10 and print how big each part is."""
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    print("train pictures:", x_train.shape, " train labels:", y_train.shape)
    print("test pictures: ", x_test.shape, " test labels: ", y_test.shape)
    return x_train, y_train, x_test, y_test

def prepare_data(pictures, labels):  # <-- NEW (whole function)
    """Scale the pixels to 0-1 and turn the labels into one-hot rows of 10."""
    pictures_ready = pictures.astype("float32") / 255
    labels_ready = to_categorical(labels, 10)
    return pictures_ready, labels_ready

def prepare_data_vgg16(pictures, labels):  # <-- NEW (whole function, at the very end)
    """Prepare the pictures the way VGG16 saw them on ImageNet, and one-hot the labels."""
    pictures_ready = preprocess_input(pictures.astype("float32"))
    labels_ready = to_categorical(labels, 10)
    return pictures_ready, labels_ready