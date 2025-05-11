from typing import Type
from .base import BaseModel
from .folk_rnn import FolkRNN
from .music_vae import MusicVae
from .gpt import GPT2
from .performance_rnn import PerformanceRNN


def get_model(name: str) -> Type[BaseModel]:
    match name:
        case "performance-rnn":
            return PerformanceRNN
        case "folk-rnn":
            return FolkRNN
        case "music-vae":
            return MusicVae
        case "gpt2":
            return GPT2
        case _:
            raise NotImplementedError(f"Model {name} is not available")
