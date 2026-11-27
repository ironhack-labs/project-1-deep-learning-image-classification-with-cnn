"""Save 10 test pictures (one per class) in a folder, to try out the web app."""
from pathlib import Path

import numpy as np
from PIL import Image

from project_tools import CLASS_NAMES, load_data

PICTURES = Path(__file__).parent / "test_pictures"
PICTURES.mkdir(exist_ok=True)

x_train, y_train, x_test, y_test = load_data()

for class_number in range(10):
    position = np.flatnonzero(y_test == class_number)[0]
    picture = Image.fromarray(x_test[position])
    picture.save(PICTURES / f"{class_number}_{CLASS_NAMES[class_number]}.png")
    print("saved:", CLASS_NAMES[class_number], "(test picture number", position, ")")