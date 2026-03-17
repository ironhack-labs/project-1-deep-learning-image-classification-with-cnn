import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD  = (0.2470, 0.2435, 0.2616)

def get_transforms():
    """
    Train: augmentación ligera (para generalizar mejor)
    Valid/Test: sin augmentación (medición limpia)
    """
    train_tfms = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
])

    eval_tfms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    return train_tfms, eval_tfms

def get_datasets(data_dir: str):
    """
    Descarga CIFAR-10 y devuelve:
    - train_full (con transform de train)
    - train_full_eval (mismo train pero con transforms de eval para valid)
    - test_set (eval transforms)
    """
    train_tfms, eval_tfms = get_transforms()

    train_full = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=train_tfms)
    train_full_eval = datasets.CIFAR10(root=data_dir, train=True, download=False, transform=eval_tfms)
    test_set = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=eval_tfms)

    return train_full, train_full_eval, test_set

def make_split_indices(n: int, valid_size: int, seed: int = 42):
    """
    Crea índices reproducibles para train/valid.
    """
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g).tolist()
    valid_idx = perm[:valid_size]
    train_idx = perm[valid_size:]
    return train_idx, valid_idx

def get_loaders(data_dir: str, batch_size: int, valid_size: int, num_workers: int, seed: int = 42):
    """
    Devuelve train_loader, valid_loader, test_loader.
    """
    train_full, train_full_eval, test_set = get_datasets(data_dir)

    train_idx, valid_idx = make_split_indices(len(train_full), valid_size=valid_size, seed=seed)

    train_set = Subset(train_full, train_idx)             
    valid_set = Subset(train_full_eval, valid_idx)          

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader