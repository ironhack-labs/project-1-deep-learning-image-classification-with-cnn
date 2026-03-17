from data import get_loaders

def main():
    train_loader, valid_loader, test_loader = get_loaders(
        data_dir="./data",
        batch_size=64,
        valid_size=5000,
        num_workers=0,  # CPU + Windows/Mac suele ir mejor así al principio
        seed=42
    )

    x, y = next(iter(train_loader))
    print("Train batch X shape:", x.shape)
    print("Train batch y shape:", y.shape)
    print("First labels:", y[:10].tolist())

    print("Train size:", len(train_loader.dataset))
    print("Valid size:", len(valid_loader.dataset))
    print("Test size:", len(test_loader.dataset))

if __name__ == "__main__":
    main()