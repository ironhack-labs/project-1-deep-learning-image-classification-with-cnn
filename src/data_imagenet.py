from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

def get_transforms_imagenet():
    train_tfms = transforms.Compose([
        transforms.Resize(128),
        transforms.RandomCrop(128, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms

def make_split_indices(n: int, valid_size: int, seed: int = 42):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    valid_idx = perm[:valid_size]
    train_idx = perm[valid_size:]
    return train_idx, valid_idx

def get_loaders_imagenet(data_dir: str, batch_size: int, valid_size: int, num_workers: int, seed: int = 42):
    train_tfms, eval_tfms = get_transforms_imagenet()

    train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    train_full_eval = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_tfms)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_tfms)

    train_idx, valid_idx = make_split_indices(len(train_full), valid_size, seed)

    train_set = Subset(train_full, train_idx)
    valid_set = Subset(train_full_eval, valid_idx)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader