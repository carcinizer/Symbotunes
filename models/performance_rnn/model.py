import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.optim import RMSprop
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..base import BaseModel


class PerformanceRNN(BaseModel):
    """"""

    def __init__(
        self,
        event_dim: int,
        control_dim: int,
        init_dim: int,
        hidden_dim: int,
        layers: int = 3,
        dropout: float = 0.3,
        *args,
        **kwargs
    ) -> None:
        super.__init__(*args, **kwargs)
        
        self.event_dim = event_dim,
        self.control_dim = control_dim
        self.init_dim = init_dim
        self.hidden_dim = hidden_dim
        self.layers = layers
        self.dropout = dropout

        self.gru = nn.GRU()
        
    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:

    

    def _step(self, batch) -> torch.Tensor:
        pass

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def configure_optimizers(self):
        pass

    @torch.no_grad()
    def sample(self, batch_size: int) -> list[torch.Tensor]:
        pass

