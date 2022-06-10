import torch_optimizer as torch_optim
from fetch_data import (
    fetch_A9A,
    fetch_CLICK,
    fetch_FOREST,
    fetch_MICROSOFT,
    fetch_YEAR,
)

DATASET_MAP = {
    "FOREST": {
        "callable": fetch_FOREST,
        "path": "datasets/forest",
        "task": "classification",
    },
    "CLICK": {
        "callable": fetch_CLICK,
        "path": "datasets/click",
        "task": "classification",
    },
    "MICROSOFT": {
        "callable": fetch_MICROSOFT,
        "path": "datasets/microsoft",
        "task": "regression",
    },
    "YEAR": {
        "callable": fetch_YEAR,
        "path": "datasets/year",
        "task": "regression",
    },
    "A9A": {
        "callable": fetch_A9A,
        "path": "datasets/a9a",
        "task": "classification",
    },
}

OPTIMIZER_MAP = {
    "adam": {"param": "Adam", "custom": False},
    "adamw": {"param": "AdamW", "custom": False},
    "rmsprop": {"param": "RMSprop", "custom": False},
    "qhadam": {"param": torch_optim.QHAdam, "custom": True},
    "ranger": {"param": torch_optim.Ranger, "custom": True},
}

LEARNING_RATE_SCHEDULER_MAP = {
    "cosine": {"param": "CosineAnnealingWarmRestarts", "args": {"T_0": 50}},
    "none": {"param": None, "args": {}},
}
