import torch
from src.model import BaselineCNN

def main():
    model = BaselineCNN(num_classes=10)
    x = torch.randn(8, 3, 32, 32)
    y = model(x)
    print("Output shape:", y.shape)

if __name__ == "__main__":
    main()