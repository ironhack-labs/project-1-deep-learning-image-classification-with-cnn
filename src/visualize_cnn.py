import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.config import Config
from src.data import get_loaders
from src.model import BaselineCNN

# Mean/Std CIFAR-10 (igual que en data.py)
CIFAR10_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)
CIFAR10_STD  = np.array([0.2470, 0.2435, 0.2616], dtype=np.float32)

CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

def unnormalize(img_chw: np.ndarray) -> np.ndarray:
    """(3,H,W) normalized -> (H,W,3) in [0,1]"""
    img = img_chw.transpose(1, 2, 0)
    img = (img * CIFAR10_STD) + CIFAR10_MEAN
    img = np.clip(img, 0.0, 1.0)
    return img

def find_first_conv(model: torch.nn.Module) -> torch.nn.Conv2d:
    """Find first Conv2d layer anywhere in the model."""
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            return m
    raise RuntimeError("No Conv2d layer found in the model.")

def save_first_layer_filters(model: torch.nn.Module, out_path: str, max_filters: int = 32):
    """
    Saves a grid of learned filters from the first Conv2d.
    For RGB conv: weights shape = (out_channels, 3, k, k)
    We normalize each filter for visualization.
    """
    conv = find_first_conv(model)
    W = conv.weight.detach().cpu().numpy()  # (F, 3, k, k)

    F = min(W.shape[0], max_filters)
    cols = 8
    rows = int(np.ceil(F / cols))

    plt.figure(figsize=(12, 2 * rows))
    for i in range(F):
        w = W[i]  # (3,k,k)
        # normalize to [0,1] for display
        w = w - w.min()
        if w.max() > 0:
            w = w / w.max()
        img = w.transpose(1, 2, 0)  # (k,k,3)

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(img)
        ax.set_title(f"F{i}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

@torch.no_grad()
def get_activations_first_conv(model: torch.nn.Module, x: torch.Tensor):
    """
    Returns activations from the first Conv2d after forward hook.
    """
    conv = find_first_conv(model)
    activations = {}

    def hook_fn(module, inp, out):
        activations["act"] = out.detach()

    handle = conv.register_forward_hook(hook_fn)
    _ = model(x)
    handle.remove()

    return activations["act"]  # shape: (B, F, H, W)

def save_feature_maps(model: torch.nn.Module, x_single: torch.Tensor, out_path: str, max_maps: int = 32):
    """
    x_single: (1,3,32,32) normalized tensor
    Saves a grid of feature maps from first conv layer.
    """
    act = get_activations_first_conv(model, x_single)  # (1,F,H,W)
    act = act[0].cpu().numpy()  # (F,H,W)

    F = min(act.shape[0], max_maps)
    cols = 8
    rows = int(np.ceil(F / cols))

    plt.figure(figsize=(12, 2 * rows))
    for i in range(F):
        fm = act[i]
        # normalize feature map for display
        fm = fm - fm.min()
        if fm.max() > 0:
            fm = fm / fm.max()

        ax = plt.subplot(rows, cols, i + 1)
        ax.imshow(fm)
        ax.set_title(f"M{i}", fontsize=10)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def main():
    cfg = Config()
    device = torch.device(cfg.device)

    os.makedirs("reports", exist_ok=True)
    os.makedirs("reports/visuals", exist_ok=True)

    # Load test loader
    _, _, test_loader = get_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        valid_size=cfg.valid_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed
    )

    # Load model + checkpoint
    model = BaselineCNN(num_classes=cfg.num_classes).to(device)
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_best.pt")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", ckpt_path)
    print("Checkpoint epoch:", checkpoint.get("epoch"))
    print("Checkpoint best_acc:", checkpoint.get("best_acc"))

    # 1) Save first-layer filters
    filters_path = "reports/visuals/first_layer_filters.png"
    save_first_layer_filters(model, filters_path, max_filters=32)
    print("Saved:", filters_path)

    # 2) Pick a few test images and save their feature maps
    x_batch, y_batch = next(iter(test_loader))
    x_batch = x_batch.to(device)

    # We'll visualize first 3 images from batch
    for idx in range(3):
        x_single = x_batch[idx:idx+1]  # (1,3,32,32)
        y_true = int(y_batch[idx].item())
        label = CIFAR10_CLASSES[y_true]

        # Save original image (unnormalized)
        orig = unnormalize(x_batch[idx].detach().cpu().numpy())
        orig_path = f"reports/visuals/sample_{idx}_original_{label}.png"
        plt.figure(figsize=(3,3))
        plt.imshow(orig)
        plt.title(f"True: {label}")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(orig_path, dpi=200)
        plt.close()
        print("Saved:", orig_path)

        # Save feature maps grid
        fmap_path = f"reports/visuals/sample_{idx}_featuremaps.png"
        save_feature_maps(model, x_single, fmap_path, max_maps=32)
        print("Saved:", fmap_path)

if __name__ == "__main__":
    main()