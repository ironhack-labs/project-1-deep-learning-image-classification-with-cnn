import os
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import classification_report, precision_recall_fscore_support, accuracy_score

from src.config import Config
from src.data_imagenet import get_loaders_imagenet
from src.model_tl import build_resnet18

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

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
        title = "Confusion Matrix (Normalized)"
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
            txt = f"{val:.2f}" if normalize else str(int(val))
            plt.text(j, i, txt, ha="center", va="center", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

@torch.no_grad()
def main():
    cfg = Config()
    device = torch.device(cfg.device)

    # IMPORTANT: loaders with ImageNet-style preprocessing (Resize 224 + ImageNet norm)
    _, _, test_loader = get_loaders_imagenet(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        valid_size=cfg.valid_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # Build TL model + load checkpoint
    model = build_resnet18(num_classes=cfg.num_classes, freeze_backbone=False).to(device)

    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_best.pt")
    print("Loading checkpoint:", ckpt_path)
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_targets = []
    total_loss = 0.0
    total = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        all_preds.append(preds.cpu().numpy())
        all_targets.append(y.cpu().numpy())

        total_loss += loss.item() * x.size(0)
        total += x.size(0)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    test_loss = total_loss / total
    acc = accuracy_score(all_targets, all_preds)

    prec_macro, rec_macro, f1_macro, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="macro", zero_division=0
    )
    prec_w, rec_w, f1_w, _ = precision_recall_fscore_support(
        all_targets, all_preds, average="weighted", zero_division=0
    )

    print("\n=== Test Metrics (Transfer Learning: ResNet18) ===")
    print(f"Loaded checkpoint: {ckpt_path}")
    print(f"Test loss: {test_loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision (macro): {prec_macro:.4f} | Recall (macro): {rec_macro:.4f} | F1 (macro): {f1_macro:.4f}")
    print(f"Precision (weighted): {prec_w:.4f} | Recall (weighted): {rec_w:.4f} | F1 (weighted): {f1_w:.4f}")

    print("\n=== Classification Report (per class) ===")
    print(classification_report(all_targets, all_preds, target_names=CIFAR10_CLASSES, digits=4, zero_division=0))

    os.makedirs("reports", exist_ok=True)
    cm = confusion_matrix(all_preds, all_targets, num_classes=cfg.num_classes)

    plot_confusion_matrix(cm, CIFAR10_CLASSES, normalize=False, out_path="reports/tl_confusion_counts.png")
    plot_confusion_matrix(cm, CIFAR10_CLASSES, normalize=True, out_path="reports/tl_confusion_normalized.png")

    print("\nSaved:")
    print(" - reports/tl_confusion_counts.png")
    print(" - reports/tl_confusion_normalized.png")

if __name__ == "__main__":
    main()