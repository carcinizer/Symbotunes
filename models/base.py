from abc import ABC, abstractmethod
from typing import Callable
import pytorch_lightning as pl
import torch



class BaseModel(pl.LightningModule, ABC):
    """Base model for the repo"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._output_transform = None
    
    @property
    def output_transform(self) -> Callable | None:
        return self._output_transform

    @output_transform.setter
    def output_transform(self, transform: Callable | None) -> None:
        self._output_transform = transform

    @abstractmethod
    def sample(self, batch_size: int) -> list[torch.Tensor] | torch.Tensor:
        """Generate new samples, by sampling from model"""

    def sample_midi(self, batch_size: int) -> list[bytes]:
        """Generate new samples and convert them into contents of a MIDI file"""

        assert self._output_transform != None
        return [self._output_transform(x) for x in self.sample(batch_size)]
