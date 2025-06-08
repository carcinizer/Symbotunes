import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from ..base import BaseModel


class PerformanceRNN(BaseModel):
    def __init__(
        self,
        num_layers: int,
        lstm_size: int,
        vocab_size: int,
        dropout: float,
        lr: float = 0.001,
        lr_decay_start: int = 0,
        lr_decay: float = 1.0,
        num_tokens_per_sample: int = 500,
        teacher_forcing_ratio: float = 1.0,
        start_token: int = 0,
        end_token: int = 1,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lstm_size = lstm_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.dropout = dropout
        self.lr = lr
        self.lr_decay_start = lr_decay_start
        self.lr_decay = lr_decay
        self.num_tokens_per_sample = num_tokens_per_sample
        self.start_token = start_token
        self.end_token = end_token

        weights_emb = torch.eye(vocab_size, dtype=torch.float32)
        self.embedding = nn.Embedding.from_pretrained(weights_emb, freeze=True)

        self.lstm = nn.LSTM(
            input_size=vocab_size,
            hidden_size=lstm_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=False
        )

        self.out = nn.Linear(lstm_size, vocab_size)
        self.loss_fn = nn.CrossEntropyLoss(reduction="none")

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        emb = self.embedding(x)  
        assert torch.all(lengths > 0), f"Found zero-length sequence: {lengths}"
        assert torch.all(lengths <= x.size(1)), f"Length(s) exceed sequence length {x.size(1)}: {lengths}"
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        h0 = torch.zeros((self.num_layers, x.size(0), self.lstm_size), device=x.device)
        c0 = torch.zeros((self.num_layers, x.size(0), self.lstm_size), device=x.device)
        packed_out, _ = self.lstm(packed, (h0, c0))
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=x.size(1))
        logits = self.out(out) 
        return logits
    
    def _step_scheduled_sampling(self, batch) -> torch.Tensor:
        x, lengths = batch
        y = x[:, 1:] 
        x_in = x[:, :-1] 
        lengths = lengths - 1

        batch_size, seq_len = x_in.size()
        device = x.device

        h = torch.zeros((self.num_layers, batch_size, self.lstm_size), device=device)
        c = torch.zeros((self.num_layers, batch_size, self.lstm_size), device=device)

        inputs = self.embedding(x_in[:, 0]).unsqueeze(1)
        logits_all = []

        for t in range(seq_len):
            output, (h, c) = self.lstm(inputs, (h, c))
            logits = self.out(output.squeeze(1)) 
            logits_all.append(logits.unsqueeze(1)) 

            use_teacher = torch.rand(batch_size, device=device) < self.teacher_forcing_ratio
            if t + 1 < seq_len:
                next_input_idx = torch.where(
                    use_teacher,
                    x_in[:, t + 1],
                    logits.argmax(dim=-1)
                )
                inputs = self.embedding(next_input_idx).unsqueeze(1)

        logits = torch.cat(logits_all, dim=1) 
        logits = logits.permute(0, 2, 1) 
        loss_per_token = self.loss_fn(logits, y)

        mask = (torch.arange(y.size(1), device=lengths.device)[None, :] < lengths[:, None]).float()
        loss = (loss_per_token * mask).sum(dim=1) / lengths
        return loss.mean()


    def _step(self, batch) -> torch.Tensor:
        x, lengths = batch  
        y = x[:, 1:] 
        x_in = x[:, :-1]  
        lengths = lengths - 1  
        logits = self(x_in, lengths.cpu())
        logits = logits.permute(0, 2, 1)  
        loss_per_token = self.loss_fn(logits, y)

        mask = (torch.arange(y.size(1), device=lengths.device)[None, :] < lengths[:, None]).float()
        loss = (loss_per_token * mask).sum(dim=1) / lengths
        return loss.mean()

    def training_step(self, batch, batch_idx):
        if self.trainer.current_epoch >= 10:
            loss = self._step_scheduled_sampling(batch)
        else:
            loss = self._step(batch)

        lr = self.optimizers().param_groups[0]["lr"]
        self.log("train/lr", lr, prog_bar=True)
        self.log("train/loss", loss, prog_bar=True)
        return loss

    def on_epoch_start(self):
        epoch = self.trainer.current_epoch
        if epoch >= 10 and epoch % 5 == 0:
            self.teacher_forcing_ratio = max(0.0, self.teacher_forcing_ratio - 0.1)
        self.log("train/teacher_forcing_ratio", self.teacher_forcing_ratio, prog_bar=True)

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val/loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(params=self.parameters(), lr=self.lr)
        scheduler = LambdaLR(
            optimizer,
            lr_lambda=lambda epoch: 1.0 if epoch < self.lr_decay_start else self.lr_decay ** (epoch - self.lr_decay_start)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    @torch.no_grad()
    def sample(self, batch_size: int, temperature: float = 1.0) -> list[torch.Tensor]:
        self.eval()
        device = self.device

        batch = torch.full((batch_size, 1), self.start_token, device=device, dtype=torch.long)
        samples = []

        h = torch.zeros((self.num_layers, batch_size, self.lstm_size), device=device)
        c = torch.zeros((self.num_layers, batch_size, self.lstm_size), device=device)

        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(self.num_tokens_per_sample):
            emb = self.embedding(batch[:, -1:])
            output, (h, c) = self.lstm(emb, (h, c))  
            logits = self.out(output.squeeze(1)) 
            probs = F.softmax(logits / temperature, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1) 

            batch = torch.cat([batch, next_tokens], dim=1)
            just_finished = (next_tokens.squeeze(1) == self.end_token) & ~finished
            for idx in just_finished.nonzero(as_tuple=False).squeeze(1):
                samples.append(batch[idx])

            finished |= just_finished
            if finished.all():
                break

            if finished.any():
                active = ~finished
                batch = batch[active]
                h = h[:, active]
                c = c[:, active]
                finished = finished[active]

        for i in range(batch.size(0)):
            samples.append(batch[i])

        return samples