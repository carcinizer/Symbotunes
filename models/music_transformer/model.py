import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel


class Norm(nn.Module):
    def __init__(self, n_state, epsilon=1e-5):
        super(Norm, self).__init__()
        self.epsilon = epsilon
        self.layer_norm = nn.LayerNorm(n_state, eps=epsilon)

    def forward(self, x):
        return self.layer_norm(x)
    
class MLP(nn.Module):
    def __init__(self, n_embd):
        super(MLP, self).__init__()
        
        self.c_fc = nn.Linear(n_embd, 4*n_embd)
        self.act = nn.GELU() 
        self.c_proj = nn.Linear(4*n_embd, n_embd)

    def forward(self, x):
        h = self.act(self.c_fc(x)) 
        h2 = self.c_proj(h)  
        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, n_embd, n_head, scale=False):
        super(Block, self).__init__()
        self.attn = MultiheadAttention(n_embd, n_ctx, n_head, scale)
        self.ln_1 = Norm(n_embd)
        self.mlp = MLP(n_embd)
        self.ln_2 = Norm(n_embd)

    def forward(self, x, layer_past=None, attn_mask=None):
        a, present = self.attn(self.ln_1(x), layer_past, attn_mask)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present
    
class MultiheadAttention(nn.Module):
    def __init__(self, nx, n_ctx, n_head, scale=False):
        super().__init__()
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.scale = scale
        self.head_dim = nx // n_head
        self.split_size = nx

        self.c_attn = nn.Linear(nx, 3 * nx)
        self.c_proj = nn.Linear(nx, nx)

        self.rel_pos_emb = nn.Embedding(2 * n_ctx - 1, self.head_dim)

        # Biasy wg Shaw et al. (Transformer-XL)
        self.u = nn.Parameter(torch.Tensor(self.n_head, self.head_dim))
        self.v = nn.Parameter(torch.Tensor(self.n_head, self.head_dim))

        self.register_buffer("bias", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.v, std=0.02)

    def split_heads(self, x, is_key=False):
        B, T, C = x.size()
        x = x.view(B, T, self.n_head, C // self.n_head)
        if is_key:
            return x.permute(0, 2, 3, 1)  # (B, H, D, T)
        else:
            return x.permute(0, 2, 1, 3)  # (B, H, T, D)

    def merge_heads(self, x):
        B, H, T, D = x.size()
        return x.permute(0, 2, 1, 3).contiguous().view(B, T, H * D)

    def _skew(self, x):
        B, H, T, _ = x.size()
        x_padded = F.pad(x, (1, 0))                # [B, H, T, 2T]
        x_padded = x_padded.view(B, H, -1, T)      # [B, H, 2T, T]
        return x_padded[:, :, T - 1:T - 1 + T]     # [B, H, T, T]


    def forward(self, x, layer_past=None, attn_mask=None):
        B, T, C = x.size()

        x = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = x.split(C, dim=2)

        q = self.split_heads(q)  # (B, H, T, D)
        k = self.split_heads(k, is_key=True)  # (B, H, D, T)
        v = self.split_heads(v)  # (B, H, T, D)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-1)
            v = torch.cat((past_v, v), dim=2)

        present = (k, v)

        # Relatywna pozycja (2T - 1)
        rel_emb = self.rel_pos_emb(torch.arange(2 * T - 1, device=x.device))  # [2T - 1, D]
        rel_emb = rel_emb.view(2 * T - 1, self.head_dim).unsqueeze(0).unsqueeze(0)  # [1, 1, 2T-1, D]

        # Składowe attention
        AC = torch.matmul(q + self.u.unsqueeze(0).unsqueeze(2), k)  # [B, H, T, T]
        BD = torch.matmul(q + self.v.unsqueeze(0).unsqueeze(2), rel_emb.transpose(-2, -1))  # [B, H, T, 2T-1]
        BD = self._skew(BD)[:, :, :, :T]  # [B, H, T, T]
        scores = AC + BD

        if self.scale:
            scores = scores / self.head_dim ** 0.5

        if attn_mask is not None:
            scores = scores + attn_mask  # Zakładamy maskę w stylu [-inf, 0]

        # Causal mask (jeśli brak `attn_mask`)
        else:
            scores = scores.masked_fill(self.bias[:, :, -T:, :T] == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)  # (B, H, T, D)

        # Dodanie relatywnego komponentu wartości (Shaw et al. rel_v)
        v_rel = torch.matmul(attn_weights, rel_emb[:, :, T-1:])  # (B, H, T, D)
        attn_output = attn_output + v_rel

        attn_output = self.merge_heads(attn_output)
        attn_output = self.c_proj(attn_output)

        return attn_output, present

class MusicTransformer(BaseModel):
    def __init__(
        self,
        n_vocab,
        n_ctx,
        n_embd,
        n_head,
        n_layer,
        lr: float = 0.003,
        lr_decay: float = 0.97,
        lr_decay_start: int = 20,
        start_token: int = 135,
        end_token: int = 136,
        temperature: float = 1.0,
        *args,
        **kwargs
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_vocab = n_vocab
        self.n_ctx = n_ctx
        self.n_embd = n_embd
        self.n_head = n_head
        self.n_layer = n_layer
        self.lr = lr
        self.lr_decay = lr_decay
        self.lr_decay_start = lr_decay_start
        self.temperature = temperature

        self.wte = nn.Embedding(self.n_vocab, self.n_embd)
        block = Block(self.n_ctx, self.n_embd, self.n_head, scale=True)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(self.n_layer)])
        self.ln_f = Norm(self.n_embd)
        n_hidden = 2 * n_vocab
        self.ll = nn.Linear(n_embd, n_hidden)
        self.ll2 = nn.Linear(n_hidden, n_vocab)

        self.start_token = start_token
        self.end_token = end_token

    def set_embeddings_weights(self, model_embeddings_weights):
        embed_shape = model_embeddings_weights.shape
        self.decoder = nn.Linear(embed_shape[1], embed_shape[0], bias=False)
        self.decoder.weight = model_embeddings_weights

    def forward(self, input_ids, position_ids=None, past=None):

        if past is None:
            past_length = 0
            past = [None] * len(self.h)
        else:
            past_length = past[0][0].size(-2)

        if position_ids is None:
            position_ids = torch.arange(
                past_length, input_ids.size(-1) + past_length, dtype=torch.long, device=input_ids.device
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))
        position_ids = position_ids.view(-1, position_ids.size(-1))

        input_embeds = self.wte(input_ids)
        hidden_states = input_embeds
        
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(*output_shape)

        out = F.leaky_relu(self.ll(hidden_states))
        out = self.ll2(out)

        return out, presents

    def _step(self, batch) -> torch.Tensor:
        x, lengths = batch
        y = x[:, 1:]
        mask = (torch.arange(x.shape[1], device=self.device).unsqueeze(0) < lengths.unsqueeze(1)).float()
        mask = mask[:, 1:]
        out, _ = self(x)
        d1, d2, d3 = out.shape
        out = out[:, :-1, :].reshape(d1 * (d2 - 1), d3)
        y = y.flatten()
        loss = F.cross_entropy(out, y, reduction="none")
        loss = loss.view(d1, d2 - 1)
        loss = loss * mask
        loss = loss.mean()
        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("train/loss", loss, prog_bar=True, logger=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._step(batch)
        self.log("val/loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)

    @torch.no_grad()
    def sample(self, batch_size: int) -> list[torch.Tensor]:
        self.eval()
        batch = torch.tensor([[self.start_token] for _ in range(batch_size)], device=self.device)
        samples: list[torch.Tensor] = []
        while batch.shape[0] > 0 and batch.shape[1] < self.n_ctx:
            out, _ = self(batch)
            logits = out[:, -1] / self.temperature
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            batch = torch.concat(tensors=(batch, next_tokens), dim=1)
            ended = batch[:, -1] == self.end_token
            samples += [sample for sample in batch[ended]]
            batch = batch[~ended]
        self.train()
        return samples
