from abc import ABC
from dataclasses import dataclass


@dataclass
class BaseModelConfig(ABC):
    """Base class for architecture configurations."""

    name: str = ""
    random_seed: int = 42

    def __post_init__(self):
        if not self.name:
            self.name = str(self)
