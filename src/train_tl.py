import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm

from src.utils import set_seed, ensure_dir, save_checkpoint
from src.data_imagenet import get_loaders_imagenet
from src.model_tl import build_resnet18
from src.config import Config

def accuracy(logits, y):
    return (logits.argmax(1) == y).float().mean().item()

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="Train", leave=False):
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item()
        acc_sum += accuracy(logits, y)
        n += 1
    return loss_sum / n, acc_sum / n

@torch.no_grad()
def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    loss_sum, acc_sum, n = 0.0, 0.0, 0
    for x, y in tqdm(loader, desc="Valid", leave=False):
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        loss_sum += loss.item()
        acc_sum += accuracy(logits, y)
        n += 1
    return loss_sum / n, acc_sum / n

def main():
    cfg = Config()
    print("RUN_NAME:", cfg.run_name)
    print("EPOCHS:", cfg.epochs)
    print("BATCH_SIZE:", cfg.batch_size)
    print("NUM_WORKERS:", cfg.num_workers)
    print("DEVICE:", cfg.device)
    set_seed(cfg.seed)

    device = torch.device(cfg.device)  # cpu in your case
    ensure_dir(cfg.ckpt_dir)

    # IMPORTANT: use a distinct run name in config before running this
    ckpt_path = os.path.join(cfg.ckpt_dir, f"{cfg.run_name}_best.pt")

    train_loader, valid_loader, _ = get_loaders_imagenet(
        data_dir=cfg.data_dir,
        batch_size=cfg.batch_size,
        valid_size=cfg.valid_size,
        num_workers=cfg.num_workers,
        seed=cfg.seed,
    )

    model = build_resnet18(num_classes=cfg.num_classes, freeze_backbone=True).to(device)

    from src.model_tl import unfreeze_layer4

    unfreeze_layer4(model)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print("Trainable params:", sum(p.numel() for p in trainable_params))
    
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)

    best_acc = 0.0

    for epoch in range(1, cfg.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        va_loss, va_acc = eval_one_epoch(model, valid_loader, criterion, device)

        print(f"Epoch {epoch:02d}/{cfg.epochs} | train {tr_loss:.4f}/{tr_acc:.4f} | valid {va_loss:.4f}/{va_acc:.4f}")

        if va_acc > best_acc:
            best_acc = va_acc
            save_checkpoint(ckpt_path, model, optimizer, epoch, best_acc)

    print(f"\nBest valid acc: {best_acc:.4f}")
    print(f"Saved: {ckpt_path}")

if __name__ == "__main__":
    main()