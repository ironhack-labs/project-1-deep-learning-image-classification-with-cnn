import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.config import Config
from src.data import get_loaders
from src.model import BaselineCNN
from src.utils import set_seed, ensure_dir, save_checkpoint

def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    n_batches = 0

    for x, y in tqdm(loader, desc="Valid", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item()
        total_acc += accuracy_from_logits(logits, y)
        n_batches += 1

    return total_loss / n_batches, total_acc / n_batches

def main():
    cfg = Config()
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    ensure_dir(cfg.ckpt_dir)
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_best.pt")

    train_loader, valid_loader, _ = get_loaders(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        valid_size=cfg.valid_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = BaselineCNN(num_classes=cfg.num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_valid_acc = 0.0
    best_epoch = 0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_one_epoch(model, valid_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"valid loss {va_loss:.4f} acc {va_acc:.4f}")

        if va_acc > best_valid_acc:
            best_valid_acc = va_acc
            best_epoch = epoch
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_valid_acc)

    print(f"\nBest valid acc: {best_valid_acc:.4f} (epoch {best_epoch})")
    print(f"Saved best checkpoint to: {ckpt_path}")

if __name__ == "__main__":
    main()