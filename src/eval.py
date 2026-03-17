import torch
import torch.nn as nn

from src.config import Config
from src.data import get_loaders
from src.model import BaselineCNN

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

@torch.no_grad()
def evaluate():
    cfg = Config()
    device = torch.device(cfg.device)

    # Cargar loaders
    _, _, test_loader = get_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        valid_size=cfg.valid_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    # Crear modelo
    model = BaselineCNN(num_classes=cfg.num_classes).to(device)

    # Cargar checkpoint
    ckpt_path = f"{cfg.ckpt_dir}/{cfg.run_name}_best.pt"
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    criterion = nn.CrossEntropyLoss()

    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    test_loss = total_loss / n_batches
    test_acc = total_acc / n_batches

    print(f"\nTest loss: {test_loss:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    evaluate()