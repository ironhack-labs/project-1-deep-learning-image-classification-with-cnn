"""Step 2: build the CNN, train it, and save the model and its learning curves."""
from pathlib import Path

import keras
import matplotlib.pyplot as plt
from keras import layers

from project_tools import load_data, prepare_data

EPOCHS = 50  # first a test run with 1 epoch to measure the time, then 50
BATCH_SIZE = 128

FOLDER = Path(__file__).parent
MODELS = FOLDER / "models"
FIGURES = FOLDER / "figures"
MODELS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def build_cnn():
    """Build the CNN: data augmentation, 3 VGG blocks, then the classifier."""
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        # data augmentation (only active during training)
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        # block 1: 32 filters
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        # block 2: 64 filters
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        # block 3: 128 filters
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), padding="same", activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        # classifier
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    return model


def plot_history(history):
    """Save and show the loss and accuracy curves of the training."""
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    ax_loss.plot(history.history["loss"], label="training")
    ax_loss.plot(history.history["val_loss"], label="validation")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("epoch")
    ax_loss.legend()
    ax_acc.plot(history.history["accuracy"], label="training")
    ax_acc.plot(history.history["val_accuracy"], label="validation")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("epoch")
    ax_acc.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "training_curves.png")
    plt.show()


keras.utils.set_random_seed(42)

x_train, y_train, x_test, y_test = load_data()
x_train, y_train = prepare_data(x_train, y_train)

model = build_cnn()
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3),
]

history = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS,
    validation_split=0.1,
    callbacks=callbacks,
)

model.save(MODELS / "cnn_cifar10.keras")
print("Model saved:", MODELS / "cnn_cifar10.keras")
plot_history(history)