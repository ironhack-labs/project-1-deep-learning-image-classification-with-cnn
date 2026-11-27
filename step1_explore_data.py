"""Step 1: load CIFAR-10, count the classes and look at the pictures."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from project_tools import CLASS_NAMES, load_data

FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)


def count_per_class(labels):
    """Print how many pictures each class has."""
    numbers, counts = np.unique(labels, return_counts=True)
    for number, count in zip(numbers, counts):
        print(CLASS_NAMES[number], count)


def show_examples(images, labels, per_class=10):
    """Save and show a grid: one row per class, random pictures in each row."""
    fig, axes = plt.subplots(10, per_class, figsize=(12, 12))
    for class_number in range(10):
        positions = np.flatnonzero(labels == class_number)
        chosen = np.random.choice(positions, per_class, replace=False)
        for column in range(per_class):
            axes[class_number, column].imshow(images[chosen[column]])
            axes[class_number, column].axis("off")
        axes[class_number, 0].set_title(CLASS_NAMES[class_number], fontsize=9, loc="left")
    plt.tight_layout()
    plt.savefig(FIGURES / "examples_grid.png")
    plt.show()


x_train, y_train, x_test, y_test = load_data()
count_per_class(y_train)
show_examples(x_train, y_train)