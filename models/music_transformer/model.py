import numpy as np
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..base import BaseModel
from typing import Optional, Tuple

<<<<<<< HEAD
class Norm(nn.Module):
    def __init__(self, n_state, epsilon=1e-5):
        super(Norm, self).__init__()
        self.epsilon = epsilon
        self.layer_norm = nn.LayerNorm(n_state, eps=epsilon)

    def forward(self, x):
        return self.layer_norm(x)
=======

class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.w = nn.Parameter(w)
        self.b = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.b, x.view(-1, x.size(-1)), self.w)
        x = x.view(*size_out)
        return x
>>>>>>> 703c86d9a593fc51e613c17383ac3fe47d5c9b45
    
class Norm(nn.Module):
    def __init__(self, n_state, epsilon=1e-5):
        super(Norm, self).__init__()
        self.epsilon = epsilon
        self.layer_norm = nn.LayerNorm(n_state, eps=epsilon)

    def forward(self, x):
        return self.layer_norm(x)
    
class MLP(nn.Module):
    def __init__(self, n_embd, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.c_fc = nn.Linear(n_embd, 4 * n_embd)
        self.act = nn.GELU()
        self.c_proj = nn.Linear(4 * n_embd, n_embd)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        h = self.act(self.c_fc(x))
        h2 = self.dropout(self.c_proj(h))
        return h2

class Block(nn.Module):
    def __init__(self, n_ctx, n_embd, n_head, scale=False, dropout_rate=0.1):
        super(Block, self).__init__()
        self.attn = MultiheadAttention(n_embd, n_ctx, n_head, scale, dropout_rate=dropout_rate)
        self.ln_1 = Norm(n_embd)
        self.mlp = MLP(n_embd, dropout_rate=dropout_rate)
        self.ln_2 = Norm(n_embd)

    def forward(self, x, layer_past=None, attn_mask=None):
        a, present = self.attn(self.ln_1(x), layer_past, attn_mask)
        x = x + a
        m = self.mlp(self.ln_2(x))
        x = x + m
        return x, present
    
class MultiheadAttention(nn.Module):
    def __init__(self, nx, n_ctx, n_head, scale=True, dropout_rate=0.1, causal=True):
        super().__init__()
        self.n_head = n_head
        self.n_ctx = n_ctx
        self.scale = scale
        self.causal = causal
        self.head_dim = nx // n_head

        self.c_attn = nn.Linear(nx, 3 * nx)
        self.c_proj = nn.Linear(nx, nx)

        self.rel_k_emb = nn.Embedding(2 * n_ctx - 1, self.head_dim)
        self.rel_v_emb = nn.Embedding(n_ctx, self.head_dim)
        self.rel_v_proj = nn.Linear(self.head_dim, self.head_dim)
        
        self.u = nn.Parameter(torch.Tensor(self.n_head, self.head_dim))
        self.v = nn.Parameter(torch.Tensor(self.n_head, self.head_dim))
        nn.init.normal_(self.u, std=0.02)
        nn.init.normal_(self.v, std=0.02)
        
        self.attn_dropout = nn.Dropout(p=dropout_rate)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.normal_(self.rel_k_emb.weight, std=0.02)
        nn.init.normal_(self.rel_v_emb.weight, std=0.02)

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
    
    def compute_qkv(self, x):
        _, _, C = x.size()
        x = self.c_attn(x)  # (B, T, 3*C)
        q, k, v = x.split(C, dim=2)
        q = self.split_heads(q)         # (B, H, T, D)
        k = self.split_heads(k, is_key=True)  # (B, H, D, T)
        v = self.split_heads(v)         # (B, H, T, D)
        return q, k, v
    
    def compute_relative_embeddings(self, T, device):
        rel_k = self.rel_k_emb(torch.arange(2 * T - 1, device=device))
        rel_k = rel_k.unsqueeze(0).unsqueeze(0)  # (1, 1, 2T-1, D)

        rel_v = self.rel_v_emb(torch.arange(T, device=device))
        rel_v = self.rel_v_proj(rel_v).unsqueeze(0).unsqueeze(0)  # (1, 1, T, D)

        return rel_k, rel_v
    
    def compute_attention_scores(self, q, k, rel_k):
        AC = torch.matmul(q + self.u.unsqueeze(0).unsqueeze(2), k)  # (B, H, T, T)
        BD = torch.matmul(q + self.v.unsqueeze(0).unsqueeze(2), rel_k.transpose(-2, -1))  # (B, H, T, 2T-1)
        BD = self._skew(BD)[:, :, :, :AC.size(-1)]  # (B, H, T, T)
        return AC + BD
    
    def apply_mask_and_softmax(self, scores, attn_mask):
        T = scores.size(-1)

        if self.causal:
            causal_mask = torch.triu(torch.ones((T, T), device=scores.device), diagonal=1).bool()
            scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        if attn_mask is not None:
            scores = scores + attn_mask

        if self.scale:
            scores = scores / (self.head_dim ** 0.5)

        attn_weights = F.softmax(scores, dim=-1)
        return self.attn_dropout(attn_weights)

    def forward(self, x, layer_past=None, attn_mask=None):
        _, T, _ = x.size()
        q, k, v = self.compute_qkv(x)

        if layer_past is not None:
            past_k, past_v = layer_past
            k = torch.cat((past_k, k), dim=-1)
            v = torch.cat((past_v, v), dim=2)

        present = (k, v)

        rel_k, rel_v = self.compute_relative_embeddings(T, x.device)
        scores = self.compute_attention_scores(q, k, rel_k)
        attn_weights = self.apply_mask_and_softmax(scores, attn_mask)

        attn_output = torch.matmul(attn_weights, v)
        v_rel = torch.matmul(attn_weights, rel_v)
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
        self.drop = nn.Dropout(p=0.1)
        block = Block(self.n_ctx, self.n_embd, self.n_head, scale=True, dropout_rate=0.1)

        self.wte = nn.Embedding(self.n_vocab, self.n_embd)
        self.h = nn.ModuleList([copy.deepcopy(block) for _ in range(self.n_layer)])
        self.ln_f = Norm(self.n_embd)
        self.output_proj = nn.Linear(n_embd, n_vocab, bias=False)
        self.output_proj.weight = self.wte.weight

        self.start_token = start_token
        self.end_token = end_token
        
    def forward(self, input_ids, past=None):
        if past is None:
            past = [None] * len(self.h)

        input_shape = input_ids.size()
        input_ids = input_ids.view(-1, input_ids.size(-1))

        input_embeds = self.drop(self.wte(input_ids))
        hidden_states = input_embeds
        
        presents = []
        for block, layer_past in zip(self.h, past):
            hidden_states, present = block(hidden_states, layer_past)
            presents.append(present)

        hidden_states = self.ln_f(hidden_states)
        output_shape = input_shape + (hidden_states.size(-1),)
        hidden_states = hidden_states.view(*output_shape)
        out = self.output_proj(hidden_states)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        def lr_lambda(step):
            if step < self.lr_decay_start:
                return step / max(1, self.lr_decay_start)
            else:
                return self.lr_decay ** (step - self.lr_decay_start)

        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            }
        }

    @torch.no_grad()
    def sample(self, batch_size: int) -> list[torch.Tensor]:
        self.eval()
        device = self.device
        batch = torch.tensor([[self.start_token]] * batch_size, device=device)
        samples: list[torch.Tensor] = []
        past: list[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None] * len(self.h)
        steps = 0

        while batch.shape[0] > 0 and batch.shape[1] < self.n_ctx and steps < self.n_ctx:
            steps += 1

            if steps == 1:
                input_ids = batch
            else:
                input_ids = batch[:, -1:].contiguous() 
                
            out, present = self(input_ids, past=past)
            logits = out[:, -1] / self.temperature
            probs = torch.softmax(logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1)

            batch = torch.cat([batch, next_tokens], dim=1)
            past = present

            ended = batch[:, -1] == self.end_token
            samples += [sample for sample in batch[ended]]
            batch = batch[~ended]
            past = [p for i, p in enumerate(past) if not ended[i]] if batch.size(0) > 0 else []

        self.train()
        return samples