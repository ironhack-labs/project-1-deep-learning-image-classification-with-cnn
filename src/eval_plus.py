import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.config import Config
from src.data import get_loaders

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

def unnormalize(img_chw: np.ndarray) -> np.ndarray:
    """
    img_chw: (3, H, W) normalizada
    devuelve img_hwc: (H, W, 3) en rango [0,1] aprox
    """
    img = img_chw.transpose(1, 2, 0)
    img = (img * CIFAR10_STD) + CIFAR10_MEAN
    img = np.clip(img, 0.0, 1.0)
    return img

@torch.no_grad()
def collect_predictions(model: torch.nn.Module, loader, device):
    all_preds = []
    all_targets = []
    all_images = [] 

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        preds = torch.argmax(logits, dim=1)

        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())
        all_images.append(x.cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    all_images = np.concatenate(all_images) 
    return all_preds, all_targets, all_images

def confusion_matrix(preds: np.ndarray, targets: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(targets, preds):
        cm[t, p] += 1
    return cm

def plot_confusion_matrix(cm: np.ndarray, class_names, normalize: bool, out_path: str):
    if normalize:
        cm_plot = cm.astype(np.float32)
        row_sums = cm_plot.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm_plot, np.maximum(row_sums, 1.0))
        title = "Confusion Matrix (Normalized by True Class)"
    else:
        cm_plot = cm
        title = "Confusion Matrix (Counts)"

    plt.figure(figsize=(10, 8))
    plt.imshow(cm_plot)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(ticks=np.arange(len(class_names)), labels=class_names, rotation=45, ha="right")
    plt.yticks(ticks=np.arange(len(class_names)), labels=class_names)

    plt.colorbar()

    for i in range(cm_plot.shape[0]):
        for j in range(cm_plot.shape[1]):
            val = cm_plot[i, j]
            if normalize:
                text = f"{val:.2f}"
            else:
                text = str(int(val))
            plt.text(j, i, text, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_accuracy_by_class(cm: np.ndarray, class_names, out_path: str):
    totals = cm.sum(axis=1)
    correct = np.diag(cm)
    acc = np.divide(correct, np.maximum(totals, 1))

    plt.figure(figsize=(10, 5))
    plt.bar(class_names, acc)
    plt.ylim(0.0, 1.0)
    plt.title("Accuracy by Class (Test)")
    plt.ylabel("Accuracy")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def save_error_grid(images_chw: np.ndarray, targets: np.ndarray, preds: np.ndarray, out_path: str, max_images: int = 25):
    wrong_idx = np.where(preds != targets)[0]
    if len(wrong_idx) == 0:
        print("No errors found (rare). Nothing to save.")
        return

    wrong_idx = wrong_idx[:max_images]

    n = len(wrong_idx)
    cols = 5
    rows = int(np.ceil(n / cols))

    plt.figure(figsize=(12, 3 * rows))
    for k, idx in enumerate(wrong_idx):
        img = unnormalize(images_chw[idx])
        true_name = CIFAR10_CLASSES[int(targets[idx])]
        pred_name = CIFAR10_CLASSES[int(preds[idx])]

        ax = plt.subplot(rows, cols, k + 1)
        ax.imshow(img)
        ax.set_title(f"T: {true_name}\nP: {pred_name}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    cfg = Config()
    device = torch.device(cfg.device)

    from src.model import BaselineCNN

    _, _, test_loader = get_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        valid_size=cfg.valid_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = BaselineCNN(num_classes=cfg.num_classes).to(device)
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    print("Loaded checkpoint:", ckpt_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint best_acc:", checkpoint.get("best_acc"))
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_correct = 0
    total = 0

    all_preds, all_targets, all_images = collect_predictions(model, test_loader, device)

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            total_loss += loss.item() * x.size(0)

    total = len(all_targets)
    total_correct = int((all_preds == all_targets).sum())
    test_loss = total_loss / total
    test_acc = total_correct / total

    print(f"\nLoaded checkpoint: {ckpt_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

    cm = confusion_matrix(all_preds, all_targets, num_classes=cfg.num_classes)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/errors", exist_ok=True)

    plot_confusion_matrix(cm, CIFAR10_CLASSES, normalize=False, out_path="reports/confusion_counts.png")
    plot_confusion_matrix(cm, CIFAR10_CLASSES, normalize=True,  out_path="reports/confusion_normalized.png")
    plot_accuracy_by_class(cm, CIFAR10_CLASSES, out_path="reports/accuracy_by_class.png")
    save_error_grid(all_images, all_targets, all_preds, out_path="reports/errors/misclassified_grid.png", max_images=25)

    print("\nSaved:")
    print(" - reports/confusion_counts.png")
    print(" - reports/confusion_normalized.png")
    print(" - reports/accuracy_by_class.png")
    print(" - reports/errors/misclassified_grid.png")

if __name__ == "__main__":
    main()