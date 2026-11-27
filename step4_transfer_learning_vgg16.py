"""Step 4: transfer learning with VGG16 (pretrained on ImageNet), then save the model."""
from pathlib import Path

import keras
import matplotlib.pyplot as plt
from keras import layers
from keras.applications import VGG16

from project_tools import load_data, prepare_data_vgg16

EPOCHS_ROUND_1 = 10  # time test first; then 10
EPOCHS_ROUND_2 = 50  # time test first; then 50 (a maximum, EarlyStopping stops earlier)
BATCH_SIZE = 128

FOLDER = Path(__file__).parent
MODELS = FOLDER / "models"
FIGURES = FOLDER / "figures"
MODELS.mkdir(exist_ok=True)
FIGURES.mkdir(exist_ok=True)


def build_vgg16_model():
    """VGG16 without its own classifier (frozen), plus my new classifier for 10 classes."""
    base = VGG16(include_top=False, weights="imagenet", input_shape=(32, 32, 3))
    base.trainable = False
    model = keras.Sequential([
        keras.Input(shape=(32, 32, 3)),
        # data augmentation (only active during training)
        layers.RandomFlip("horizontal"),
        layers.RandomTranslation(0.1, 0.1),
        # the pretrained part: 13 convolution layers that learned on ImageNet
        base,
        # my new classifier
        layers.GlobalAveragePooling2D(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(10, activation="softmax"),
    ])
    return model, base


def plot_history(history_1, history_2):
    """Save and show loss and accuracy of both rounds; the dotted line = start of round 2."""
    round_2_starts = len(history_1.history["loss"]) - 0.5
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))
    for ax, name, title in [(ax_loss, "loss", "Loss"), (ax_acc, "accuracy", "Accuracy")]:
        ax.plot(history_1.history[name] + history_2.history[name], label="training")
        ax.plot(history_1.history["val_" + name] + history_2.history["val_" + name],
                label="validation")
        ax.axvline(round_2_starts, color="grey", linestyle=":")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.legend()
    plt.tight_layout()
    plt.savefig(FIGURES / "vgg16_training_curves.png")
    plt.show()


keras.utils.set_random_seed(42)

x_train, y_train, x_test, y_test = load_data()
x_train, y_train = prepare_data_vgg16(x_train, y_train)

model, base = build_vgg16_model()

# Round 1: VGG16 stays frozen, only my new classifier learns
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
history_1 = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_ROUND_1,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                             restore_best_weights=True)],
)

# Round 2 (fine-tuning): unfreeze VGG16 and continue with a very small learning rate
base.trainable = True
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])
model.summary()
history_2 = model.fit(
    x_train, y_train,
    batch_size=BATCH_SIZE,
    epochs=EPOCHS_ROUND_2,
    validation_split=0.1,
    callbacks=[
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ],
)

# Save without the optimizer's notes: about 60 MB instead of 178 MB
# (GitHub refuses files bigger than 100 MB)
model.compile(loss="categorical_crossentropy", metrics=["accuracy"])
model.save(MODELS / "vgg16_cifar10.keras")
print("Model saved:", MODELS / "vgg16_cifar10.keras")
plot_history(history_1, history_2)