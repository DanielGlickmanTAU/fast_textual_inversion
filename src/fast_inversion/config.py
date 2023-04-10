from dataclasses import dataclass


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 100

    exp_name: str = 'fast_inversion_train'
    use_wandb: bool = False

    validate_loss: bool = True
    log_images_every_n_epochs: int = 1
    num_persons_images_to_log: int = 1
    num_images_per_person_to_log: int = 2
