from pathlib import Path
from dataclasses import dataclass
from typing import Optional

# Defining paths
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

DATA_DIR = ROOT_DIR / "data"

@dataclass
class TrainerConfig:
    """Trainer config class."""

    ckpt: str
    dataset: str
    use_augmented_data: bool
    sampling: str
    augmentation_model: str
    wandb_project: str
    wandb_entity: str
    batch_size: int
    num_epochs: int
    lr: float
    weight_decay: float
    experiment_type: Optional[str] = None


@dataclass
class PromptConfig:
    """Config for prompting classification."""

    model: str
    dataset: str


@dataclass
class AugmentConfig:
    """Config for prompting augmentation."""

    model: str
    dataset: str
    sampling: str


@dataclass
class SetfitParams:
    """Setfit parameters."""

    batch_size: int
    lr_body: float
    lr_head: float
    num_iterations: int
    num_epochs_body: int
    num_epochs_head: int
    weight_decay: float
    ckpt: str
    text_selection: str
    wandb_project: str
    wandb_entity: str
    experiment_type: str
    sampling: str
    augmentation_model: str
    dataset: str


HF_HUB_MODELS = {
    "llama-2-70b": "meta-llama/Llama-2-70b-chat-hf",
    "llama-2-13b": "meta-llama/Llama-2-13b-chat-hf",
    "llama-2-7b": "meta-llama/Llama-2-7b-chat-hf",
    "llama-3.3-70b": "meta-llama/Llama-3.3-70B-Instruct",
    "llama-3.1-8b": "meta-llama/Llama-3.1-8B-Instruct",

}
