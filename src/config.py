from dataclasses import dataclass

@dataclass
class Config:
    seed: int = 42
    data_dir: str = "./data"
    num_classes: int = 10

    # training
    epochs: int = 6          # CPU friendly (luego subimos)
    batch_size: int = 16     # si tu Mac sufre, baja a 64
    lr: float = 1e-3
    weight_decay: float = 1e-4

    # split
    valid_size: int = 5000

    # system
    num_workers: int = 0      # en Mac a veces 0 va mejor al inicio
    device: str = "cpu"       # tú NO tienes CUDA

    # outputs
    run_name: str = "tl_resnet18_128_layer4"
    ckpt_dir: str = "./checkpoints"