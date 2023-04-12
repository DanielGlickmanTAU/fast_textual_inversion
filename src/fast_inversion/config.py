from dataclasses import dataclass

cfg = None


@dataclass
class TrainConfig:
    # train config
    batch_size: int = 32
    learning_rate: float = 3e-4
    epochs: int = 100
    max_images_per_instance: int = 100

    # experiment config
    exp_name: str = 'fast_inversion_train'
    use_wandb: bool = False
    # experiment logging
    validate_loss: bool = True
    log_images_every_n_epochs: int = 1
    num_persons_images_to_log: int = 1
    num_images_per_person_to_log: int = 2
    validate_images_on_cpu: bool = False

    # model config
    model_type: str = 'simple'  # simplecorss


def set_config(cfg_):
    global cfg
    cfg = cfg_


def get_config() -> TrainConfig:
    return cfg
