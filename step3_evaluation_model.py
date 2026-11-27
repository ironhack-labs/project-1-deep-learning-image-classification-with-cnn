"""Step 3: load the saved CNN and evaluate it on the 10,000 test pictures."""
from pathlib import Path

import keras
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix

from project_tools import CLASS_NAMES, load_data, prepare_data

FOLDER = Path(__file__).parent
MODEL_FILE = FOLDER / "models" / "cnn_cifar10.keras"
FIGURES = FOLDER / "figures"


def show_confusion_matrix(true_classes, predicted_classes):
    """Save and show the confusion matrix with the class names."""
    cm = confusion_matrix(true_classes, predicted_classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(10, 10))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title("CNN: confusion matrix (10,000 test pictures)")
    plt.tight_layout()
    plt.savefig(FIGURES / "confusion_matrix.png")
    plt.show()


def show_wrong_guesses(pictures, true_classes, predicted_classes):
    """Save and show the first 10 test pictures the model got wrong."""
    wrong = np.flatnonzero(true_classes != predicted_classes)[:10]
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))
    for ax, position in zip(axes.flatten(), wrong):
        ax.imshow(pictures[position])
        true_name = CLASS_NAMES[true_classes[position]]
        guess_name = CLASS_NAMES[predicted_classes[position]]
        ax.set_title(f"true: {true_name}\nguess: {guess_name}", fontsize=9)
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(FIGURES / "wrong_guesses.png")
    plt.show()


x_train, y_train, x_test, y_test = load_data()
x_test_ready, y_test_ready = prepare_data(x_test, y_test)

model = keras.models.load_model(MODEL_FILE)
test_loss, test_accuracy = model.evaluate(x_test_ready, y_test_ready, verbose=0)
print("Test accuracy:", round(test_accuracy, 4))

probabilities = model.predict(x_test_ready, verbose=0)
predicted_classes = np.argmax(probabilities, axis=1)
true_classes = y_test.flatten()

report = classification_report(true_classes, predicted_classes,
                               target_names=CLASS_NAMES, digits=3)
print(report)
(FIGURES / "classification_report.txt").write_text(report)

show_confusion_matrix(true_classes, predicted_classes)
show_wrong_guesses(x_test, true_classes, predicted_classes)